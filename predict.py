#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
모델은 카테고리에 대한 확률을 예측합니다.
그런데 틀린 예측에 대해서도 (상위 카테고리가 일치하면) 점수를 받을 수 있습니다.
이 스크립트는 기대 점수를 최대화하는 카테고리를 예측하도록 개선합니다.
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import decoding
from tensor2tensor.utils import data_reader
from tensor2tensor.data_generators import problem as t2t_problem
import tensorflow as tf
from tqdm.auto import tqdm

from shopping.data.datasets import shopping


def label_compatibility(high, low):
    """x 는 y 의 상위 카테고리인가?

    :param x: tuple, label
    :param y: tuple, label
    :returns: compatibility
    :rtype: bool

    """
    # 대분류와 중분류는 -1 이 없다.
    if high[0] != low[0]:
        return False
    if high[1] != low[1]:
        return False
    if high[2] == -1:  # e.g. (1,2,-1,-1)
        return low[2] > -1
    if high[2] != low[2]:
        return False
    if high[3] == -1:  # e.g. (1,2,3,-1)
        return low[3] > -1
    return False  # x is fully defined


def compute_descendant_matrix(data_dir):
    id_to_label, _ = shopping.load_labels(data_dir, narrow=False)
    num_classes = len(id_to_label)
    score_matrix = np.array([[
        label_compatibility(id_to_label[x], id_to_label[y]) for y in range(num_classes)
    ] for x in tqdm(range(num_classes), desc="Computing compatibility matrix")])
    return score_matrix


class SequentialGreedy:
    def __init__(self, data_dir):
        self.id_to_label, _ = shopping.load_labels(data_dir, narrow=False)
        self.descendent_matrix = compute_descendant_matrix(data_dir)

    def class_id_to_label(self, class_id):
        return self.id_to_label[class_id]

    def predict_label_by_logit(self, logits):
        """Refine label predictions

        :param logits: np.ndarray
        :returns: label
        :rtype: tuple

        """
        prob = shopping.softmax(logits) + 1e-7
        descendent_mask = np.ones_like(logits)
        while descendent_mask.any():
            class_id = np.argmax(prob * descendent_mask)
            descendent_mask = self.descendent_matrix[class_id]
        return self.class_id_to_label(class_id)

    def __call__(self, prediction):
        logits = prediction["logits"].ravel()
        label = self.predict_label_by_logit(logits)
        return label


class PostProcessor:
    def __init__(self, label_predictor):
        self.label_predictor = label_predictor

    def __call__(self, prediction):
        pid = prediction["example_id"].decode("utf-8")
        label = self.label_predictor(prediction)
        return pid, label


def is_not_padding_example(prediction):
    pid = prediction["example_id"].decode("utf-8")
    return bool(pid)


def predict_input_fn(problem,
                     hparams,
                     data_dir=None,
                     params=None,
                     config=None,
                     dataset_kwargs=None):
    """Builds input pipeline for problem.

    Args:
      hparams: HParams, model hparams
      data_dir: str, data directory; if None, will use hparams.data_dir
      params: dict, may include "batch_size"
      config: RunConfig; should have the data_parallelism attribute if not using
        TPU
      dataset_kwargs: dict, if passed, will pass as kwargs to self.dataset
        method when called

    Returns:
      (features_dict<str name, Tensor feature>, Tensor targets)
    """
    mode = tf.estimator.ModeKeys.PREDICT
    partition_id, num_partitions = problem._dataset_partition(mode, config)

    num_threads = 1

    if config and hasattr(config, "data_parallelism") and config.data_parallelism:
        num_shards = config.data_parallelism.n
    else:
        num_shards = 1

    assert num_shards == 1, "Use a single datashard (i.e. 1 GPU) for prediction."

    def define_shapes(example):
        batch_size = config and config.use_tpu and params["batch_size"]
        return t2t_problem.standardize_shapes(example, batch_size=batch_size)

    # Read and preprocess
    data_dir = data_dir or (hasattr(hparams, "data_dir") and hparams.data_dir)

    dataset_kwargs = dataset_kwargs or {}
    dataset_kwargs.update({
        "mode": mode,
        "data_dir": data_dir,
        "num_threads": num_threads,
        "hparams": hparams,
        "partition_id": partition_id,
        "num_partitions": num_partitions,
    })

    dataset = problem.dataset(**dataset_kwargs)

    dataset = dataset.map(
        data_reader.cast_ints_to_int32, num_parallel_calls=num_threads)

    # Batching
    assert hparams.use_fixed_batch_size, "Use fixed examples per datashard"
    # Batch size means examples per datashard.
    dataset = dataset.apply(
        tf.contrib.data.bucket_by_sequence_length(data_reader.example_length, [],
                                                  [hparams.batch_size]))

    dataset = dataset.map(define_shapes, num_parallel_calls=num_threads)

    def prepare_for_output(example):
        if mode == tf.estimator.ModeKeys.PREDICT:
            example["infer_targets"] = example.pop("targets")
            return example
        else:
            return example, example["targets"]

    dataset = dataset.map(prepare_for_output, num_parallel_calls=num_threads)
    dataset = dataset.prefetch(2)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # This is because of a bug in the Estimator that short-circuits prediction
        # if it doesn't see a QueueRunner. DummyQueueRunner implements the
        # minimal expected interface but does nothing.
        tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, data_reader.DummyQueueRunner())

    return dataset


def make_estimator_predict_input_fn(problem,
                                    hparams,
                                    data_dir=None,
                                    dataset_kwargs=None):
    """Return predict_input_fn wrapped for Estimator."""

    def estimator_input_fn(params, config):
        return predict_input_fn(
            problem,
            hparams,
            data_dir=data_dir,
            params=params,
            config=config,
            dataset_kwargs=dataset_kwargs,
        )

    return estimator_input_fn


def main():
    parser = argparse.ArgumentParser(description="Predict Shopping Categories")

    ## Optional arguments
    parser.add_argument("--problem", type=str, help="Problem name.")

    parser.add_argument("--model", type=str, help="Which model to use.")

    parser.add_argument("--hparams_set", type=str, help="Which parameters to use.")

    parser.add_argument(
        "--checkpoint_path", type=str, help="Path to the model checkpoint.")

    parser.add_argument("--data_dir", type=str, help="Directory with training data.")

    parser.add_argument(
        "--output_file", default="predict.tsv", type=str, help="Prediction TSV")

    parser.add_argument(
        "--greedy",
        default=False,
        dest='debug',
        action='store_true',
        help="Greedy Prediction")
    parser.add_argument(
        "--debug",
        default=False,
        dest='debug',
        action='store_true',
        help="No Multiprocessing")

    args = parser.parse_args()

    # Create hparams and the model
    hparams_set = args.hparams_set
    data_dir = os.path.expanduser(args.data_dir)
    problem_name = args.problem
    hparams = trainer_lib.create_hparams(
        hparams_set, data_dir=data_dir, problem_name=problem_name)

    tokens_per_example = 200
    hparams.use_fixed_batch_size = True
    hparams.batch_size = hparams.batch_size // tokens_per_example

    model_name = args.model
    decode_hp = decoding.decode_hparams()
    run_config = trainer_lib.create_run_config(model_name)
    estimator = trainer_lib.create_estimator(
        model_name, hparams, run_config, decode_hparams=decode_hp, use_tpu=False)

    if problem_name == "shopping_public_lb":
        problem = shopping.ShoppingPublicLB()
        hierarchical = False
    if problem_name == "shopping_private_lb":
        problem = shopping.ShoppingPrivateLB()
        hierarchical = False
    if problem_name == "hierarchical_shopping_public_lb":
        problem = shopping.HierarchicalShoppingPublicLB()
        hierarchical = True
    if problem_name == "hierarchical_shopping_private_lb":
        problem = shopping.HierarchicalShoppingPrivateLB()
        hierarchical = True

    dataset_kwargs = {"shard": None, "dataset_split": "test", "max_records": -1}
    infer_input_fn = make_estimator_predict_input_fn(
        problem, hparams, dataset_kwargs=dataset_kwargs)

    checkpoint_path = os.path.expanduser(args.checkpoint_path)
    predictions = estimator.predict(infer_input_fn, checkpoint_path=checkpoint_path)

    if hierarchical:
        post_processor = PostProcessor(shopping.HierarchicalArgMax(data_dir))
    else:
        if args.greedy:
            post_processor = PostProcessor(SequentialGreedy(data_dir))
        else:
            post_processor = PostProcessor(shopping.ScoreMax(data_dir))

    predictions = filter(is_not_padding_example, predictions)
    with Pool() as p:
        if args.debug:
            predictions = map(post_processor, predictions)
        else:
            predictions = p.imap(post_processor, predictions, chunksize=1000)
        output_filepath = os.path.expanduser(args.output_file)
        with open(output_filepath, "w") as output_file:
            for pid, label in tqdm(predictions):
                output_file.write("\t".join([pid] + list(map(str, label))) + "\n")
                output_file.flush()


if __name__ == "__main__":
    main()
