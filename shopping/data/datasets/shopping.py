"""Data generators for Shopping Classification Problem"""

from collections import OrderedDict, defaultdict
from enum import Enum
import itertools
import json
import logging
import os
import random
import re
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
from tqdm.auto import tqdm
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.layers import modalities
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics
import tensorflow as tf

from shopping.data.tokenizer import SentencePieceTokenizer
from shopping.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

_IMAGE_FEATURE_SIZE = 2048
_DEFAULT_PRICE = 100000

CATEGORY_LEVELS = ["b", "m", "s", "d"]

re_sc = re.compile(r"[\!@#$%\^&\*\(\)-=\[\]\{\}\.,/\?~\+'\"|]")
re_model = re.compile(r"[a-z0-9]+[0-9-_]+[a-z0-9]*", flags=re.IGNORECASE)

SCORE_WEIGHTS = [x / 4 for x in [1, 1.2, 1.3, 1.4]]
NUM_LEVEL_CLASSES = [57, 552, 3190, 404]
CLASS_VOCAB_SIZES = [58, 553, 3191, 405]


def remove_tabs(s):
    return " ".join(s.split())


def normalize(text):
    text = re_sc.sub(" ", text)
    text = remove_tabs(text)
    text = text.lower()
    return text


def decode_utf8_generator(it):
    return (s.decode("utf-8") for s in it)


class FeatureKeys(Enum):
    ID = "pid"
    TITLE = "product"
    MODEL = "model"
    IMAGE = "img_feat"
    BRAND = "brand"
    MAKER = "maker"
    PRICE = "price"
    TIME = "updttm"


class DatasetReader(object):
    def __init__(self, file_paths, div=None):
        def get_div(file_path):
            if div is None:
                return os.path.basename(file_path).split(".")[0]
            return div

        self.files = OrderedDict([(file_path, h5py.File(file_path,
                                                        "r")[get_div(file_path)])
                                  for file_path in file_paths])

    def column_generator(self, key):
        if isinstance(key, FeatureKeys):
            key = key.value
        iterables = [(value for value in h[key]) for h in self.files.values()]
        return itertools.chain(*iterables)

    def text_column_generator(self, key):
        return decode_utf8_generator(self.column_generator(key))

    def label_generator(self):
        return zip(
            *[self.column_generator(f"{level}cateid") for level in CATEGORY_LEVELS])

    def size(self):
        return sum(h[FeatureKeys.ID.value].shape[0] for h in self.files.values())


def narrow_label(label):
    b, m, s, d = label
    if b == 57:  #성인
        b = 5  #언더웨어
    if m == 552:  #전통주/주류만들기
        m = 369  #탄산/이온음료
    if s == 3190:  #부추
        s = 1263  #호박즙/야채즙
    if d == 404:  #용인시
        d = 227  #숙박
    return (b, m, s, d)


def load_labels(data_dir, narrow):
    filepath = os.path.join(data_dir, "labels.json")
    with tf.gfile.Open(filepath) as label_file:
        id_to_label = json.load(label_file)
        label_to_id = {tuple(value): key for key, value in enumerate(id_to_label)}
    id_to_label = list(map(tuple, id_to_label))
    if narrow:
        id_to_label = list(map(narrow_label, id_to_label))
    return id_to_label, label_to_id


def load_subclasses(data_dir, narrow):
    id_to_label, _ = load_labels(data_dir, narrow=narrow)
    subclasses = defaultdict(set)
    for label in id_to_label:
        for level in range(len(label)):
            subclass_id = label[level]
            # Check unknown (padding) class
            if subclass_id > 0:
                subclasses[label[:level]].add(subclass_id)
    return subclasses


class LabelVocabulary:
    def __init__(self, data_dir, narrow):
        self._id_to_label, self._label_to_id = load_labels(data_dir, narrow)

    def id_to_label(self, idx):
        return self._id_to_label[idx]

    def label_to_id(self, label):
        return self._label_to_id[label]


class LabelDecoder:
    def __init__(self, data_dir):
        with tf.gfile.Open(os.path.join(data_dir, "category.json")) as f:
            category_ids = json.load(f)
        self.label_to_name = {
            level: dict(zip(category_ids[level].values(), category_ids[level].keys()))
            for level in CATEGORY_LEVELS
        }

    def decode_label(self, label):
        return " ▻ ".join(
            self.label_to_name[level][i] for level, i in zip(CATEGORY_LEVELS, label))


def _load_dev_ids(tmp_dir):
    id_filepath = os.path.join(tmp_dir, "id.dev.txt")
    with open(id_filepath) as f:
        dev_ids = f.read().splitlines()
    return set(dev_ids)


def _parse_time(s):
    return [int(s[i:(i + 2)]) for i in range(2, 14, 2)]


def weighted_label_accuracy(x, y):
    """x 를 선택했는데 y 가 ground truth 일 경우받게 되는 점수

    :param x: tuple, label
    :param y: tuple, label
    :returns: score
    :rtype: float

    """
    w = SCORE_WEIGHTS
    score = 0
    score += w[0] * (x[0] == y[0] > -1)
    score += w[1] * (x[1] == y[1] > -1)
    score += w[2] * (x[2] == y[2] > -1)
    score += w[3] * (x[3] == y[3] > -1)
    return score


def compute_score_matrix(data_dir):
    id_to_label, _ = load_labels(data_dir, narrow=False)
    num_classes = len(id_to_label)
    score_matrix = np.array([[
        weighted_label_accuracy(id_to_label[x], id_to_label[y])
        for y in range(num_classes)
    ] for x in tqdm(range(num_classes), desc="Computing score matrix")])
    return score_matrix


def _randomly_rotate_text(text):
    words = text.split()
    n = len(words)
    if n < 2:
        return text
    pos = random.randint(1, n - 1)
    return " ".join(words[pos:] + words[:pos])


def _remove_model_numbers(text):
    text = re_model.sub(" _ ", text)
    return text


def _augment_text(text):
    text = _randomly_rotate_text(text)
    text = _remove_model_numbers(text)
    return text


class ShoppingScore:
    def __init__(self, data_dir, name="shopping_score"):
        self.score_matrix = compute_score_matrix(data_dir)
        self.name = name

    def select_class_by_logit(self, logits):
        outputs = tf.to_int32(tf.argmax(logits, axis=-1))
        return outputs

    def __call__(self, predictions, labels, weights_fn=common_layers.weights_nonzero):
        """Class ID 를 대/중/소/세 분류 Label 로 바꿔서 가중 점수를 계산합니다."""

        print("** {} **".format(self.name))
        print(" - predictions: {}".format(predictions.shape))
        print(" - labels: {}".format(labels.shape))

        # Score Matrix = Aij 는 prediction == i, label == j 일 경우 받게되는
        # 점수를 나타냅니다.

        score_matrix = tf.constant(self.score_matrix)

        with tf.variable_scope(self.name, values=[predictions, labels]):
            padded_predictions, padded_labels = common_layers.pad_with_zeros(
                predictions, labels)
            weights = weights_fn(padded_labels)
            outputs = self.select_class_by_logit(padded_predictions)
            padded_labels = tf.to_int32(padded_labels)
            indices = tf.stack([outputs, padded_labels], axis=-1)
            return tf.gather_nd(score_matrix, indices), weights


class RefinedShoppingScore(ShoppingScore):
    """기대 Score 가 가장 큰 Class ID 를 선택할 때 받을 수 있는 점수를 계산합니다."""

    def __init__(self, data_dir, name="refined_shopping_score"):
        super().__init__(data_dir, name)

    def select_class_by_logit(self, logits):
        padded_probs = tf.nn.softmax(logits)
        padded_probs = tf.squeeze(padded_probs, axis=[1, 2, 3])
        outputs = tf.to_int32(tf.argmax(padded_probs @ self.score_matrix, axis=-1))
        outputs = common_layers.expand_squeeze_to_nd(outputs, 4, expand_dim=1)
        return outputs


class LogRealModality(modalities.RealModality):
    def bottom(self, x):
        print("** LogRealModality: bottom(x) **")
        print(" - x: {}, {}".format(x.shape, x.dtype))
        with tf.variable_scope("log_real"):
            x = tf.log(x + 1e-7)
            return tf.layers.dense(
                tf.to_float(x), self._body_input_depth, name="bottom")


class ShoppingTokenTextEncoder(text_encoder.TextEncoder):
    """A `TokenEncoder` encode/decode string tokens to/from integer ids."""

    def __init__(self, tokenizer, vocabulary):
        super().__init__()
        self._tokenizer = tokenizer
        self._vocabulary = vocabulary

    def encode(self, s):
        """Converts a space-separated string of tokens to a list of ids."""
        tokens = [token.text for token in self._tokenizer.tokenize(s)]
        ret = self.encode_list(tokens)
        return ret

    def encode_list(self, tokens):
        return [self._vocabulary.token_to_id(tok) for tok in tokens]

    def decode(self, ids, strip_extraneous=False):
        """Transform a sequence of int ids into a human-readable string."""
        return self._tokenizer.detokenize(self.decode_list(ids))

    def decode_list(self, ids):
        """Transform a sequence of int ids into a their string versions."""
        seq = ids
        return [self._vocabulary.id_to_token(i) for i in seq]

    @property
    def vocab_size(self):
        return self._vocabulary.vocab_size


@registry.register_problem
class Shopping(text_problems.Text2ClassProblem):
    """
    Shopping product classification problem.

    Put your training, evaluation, and test data in the following files in tmp_dir:
      - tmp_dir/train.chunk.01, ...
      - tmp_dir/dev.chunk.01
      - tmp_dir/test.chunk.01, ...

    Put your metadata in the following files:
      - tmp_dir/id.dev.txt
      - data_dir/labels.json
      - data_dir/category.json

    Put your SentencePiece model in the following files in tmp_dir:
      - data_dir/sentpiece.model
    """

    TRAIN_FILES = ["train.chunk.%02d" % i for i in range(1, 10)]
    EVAL_FILES = ["dev.chunk.01"]
    TEST_FILES = ["test.chunk.01", "test.chunk.02"]

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split, input_file,
                                 augment):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split, input_file,
                                          augment)
        encoder = self.get_or_create_vocab(data_dir, tmp_dir)
        sep_token_id, = encoder.encode_list([text_encoder.EOS])
        for sample in generator:
            inputs = []
            for inp in sample["inputs"]:
                inp = normalize(inp)
                inputs += encoder.encode(inp)
                inputs.append(sep_token_id)
            image = sample["image"].tolist()
            assert len(image) == _IMAGE_FEATURE_SIZE
            price = sample["price"].item()
            price = price if price > -1 else _DEFAULT_PRICE
            time_parts = _parse_time(sample["time"])
            label = sample["label"]
            example_id = sample["pid"]
            yield {
                "inputs": inputs,
                "image": image,
                "price": [price],
                "time": time_parts,
                "targets": [label],
                "example_id": [example_id],
            }

    @property
    def already_shuffled(self):
        return False

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        """Generates training/dev data.

        :param data_dir: a string
        :param tmp_dir: a string
        :param task_id: an optional integer
        :returns: shard or shards for which data was generated.
        :rtype: NoneType

        """
        logger.info("generate_data task_id=%s" % task_id)

        # task_id 는 DatasetSplit 과 상관 없이 전체 데이터셋에 대해 부여됩니다.
        # 따라서 task_id 를 적절하게 DatasetSplit 에 나누어야 합니다.

        assert self.is_generate_per_split

        assert 0 <= task_id < self.num_generate_tasks

        # 공개된 데이터셋은 Train, Dev, Test 로 나뉘어 있지만,
        # Dev 는 Public Leaderboard 를 위한 데이터셋이므로 Label 이 없습니다.

        input_file = self._task_id_to_input_file(tmp_dir, task_id)
        dataset_split, idx, augment = self._task_id_to_output_split(task_id)

        if dataset_split["split"] == problem.DatasetSplit.TRAIN:
            filepath_fn = self.training_filepaths
        elif dataset_split["split"] == problem.DatasetSplit.EVAL:
            filepath_fn = self.dev_filepaths
        elif dataset_split["split"] == problem.DatasetSplit.TEST:
            filepath_fn = self.test_filepaths

        output_file = filepath_fn(
            data_dir, dataset_split["shards"], shuffled=self.already_shuffled)[idx]

        # Which output split is this task writing to?
        split = dataset_split["split"]

        # Actually generate examples.
        generator_utils.generate_files(
            self._maybe_pack_examples(
                self.generate_encoded_samples(
                    data_dir, tmp_dir, split, input_file, augment=augment)),
            [output_file],
        )

        # Shuffle the output.
        if not self.already_shuffled:
            generator_utils.shuffle_dataset([output_file])

    @property
    def augmentation_multiplier(self):
        """1 means no augmentation."""
        return 8

    def _task_id_to_input_file(self, tmp_dir, task_id):
        idx = task_id % len(self.TRAIN_FILES)
        input_file = os.path.join(tmp_dir, self.TRAIN_FILES[idx])
        return input_file

    def _task_id_to_output_split(self, task_id):
        num_train_tasks = len(self.TRAIN_FILES) * self.augmentation_multiplier
        if task_id < num_train_tasks:
            dataset_split = self.dataset_splits[0]
            assert dataset_split["split"] == problem.DatasetSplit.TRAIN
            idx = task_id
            augment = task_id >= len(self.TRAIN_FILES)
        else:
            dataset_split = self.dataset_splits[1]
            assert dataset_split["split"] == problem.DatasetSplit.EVAL
            idx = task_id - num_train_tasks
            augment = False
        return dataset_split, idx, augment

    @property
    def multiprocess_generate(self):
        """Whether to generate the data in multiple parallel processes."""
        return True

    @property
    def num_generate_tasks(self):
        """Needed if multiprocess_generate is True."""
        # TRAIN/EVAL/TEST shards 수의 합과 tasks 수는 같다.
        return sum(split["shards"] for split in self.dataset_splits)

    def prepare_to_generate(self, data_dir, tmp_dir):
        """Make sure that the data is prepared and the vocab is generated."""
        self.get_or_create_vocab(data_dir, tmp_dir)

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [
            {
                "split": problem.DatasetSplit.TRAIN,
                "shards": len(self.TRAIN_FILES) * self.augmentation_multiplier,
            },
            {
                "split": problem.DatasetSplit.EVAL,
                "shards": len(self.TRAIN_FILES)
            },
        ]

    @property
    def is_generate_per_split(self):
        # train/eval/test 를 별도로 생성
        return True

    @property
    def vocab_type(self):
        return NotImplemented

    @property
    def oov_token(self):
        return "<UNK>"

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        # SentencePiece model 에 포함된 vocab file 을 사용합니다.
        with NamedTemporaryFile() as model_file:
            sp_model_path = os.path.join(data_dir, "sentpiece.model")
            # SentencePiece 는 GCS 를 지원하지 않기 때문에 local 에 복사합니다.
            tf.gfile.Copy(sp_model_path, model_file.name, overwrite=True)
            tokenizer = SentencePieceTokenizer(
                model_file.name, remove_space_symbol=True)
            vocabulary = Vocabulary.from_sentence_piece(
                model_file.name, oov_token=self.oov_token, remove_space_symbol=True)
        encoder = ShoppingTokenTextEncoder(tokenizer, vocabulary)
        return encoder

    def generate_samples(self, data_dir, tmp_dir, dataset_split, input_file, augment):
        files = [input_file]
        train_ds = DatasetReader(files)

        _, label_to_id = load_labels(data_dir, narrow=False)
        dev_ids = _load_dev_ids(tmp_dir)

        return self.dataset_iterator(train_ds, dataset_split, label_to_id, dev_ids,
                                     augment)

    def dataset_iterator(self, ds, dataset_split, label_to_id, dev_ids, augment):
        id_it = ds.text_column_generator(FeatureKeys.ID)
        title_it = ds.text_column_generator(FeatureKeys.TITLE)
        model_it = ds.text_column_generator(FeatureKeys.MODEL)
        image_it = ds.column_generator(FeatureKeys.IMAGE)
        brand_it = ds.text_column_generator(FeatureKeys.BRAND)
        maker_it = ds.text_column_generator(FeatureKeys.MAKER)
        price_it = ds.column_generator(FeatureKeys.PRICE)
        time_it = ds.column_generator(FeatureKeys.TIME)
        if dataset_split != problem.DatasetSplit.TEST:
            label_it = map(label_to_id.get, ds.label_generator())
        else:
            label_it = itertools.repeat(0)
        it = zip(
            id_it,
            title_it,
            model_it,
            image_it,
            brand_it,
            maker_it,
            price_it,
            time_it,
            label_it,
        )

        for pid, title, model, image, brand, maker, price, time, label_id in it:
            if dataset_split == problem.DatasetSplit.TRAIN:
                if pid in dev_ids:
                    continue
            if dataset_split == problem.DatasetSplit.EVAL:
                if pid not in dev_ids:
                    continue
            if augment:
                title = _augment_text(title)
                model = _augment_text(model)
            yield {
                "inputs": [title, model, maker, brand],
                "image": image,
                "price": price,
                "time": time,
                "label": label_id,
                "pid": pid,
            }

    @property
    def num_classes(self):
        """The number of classes."""
        return 4215

    def class_labels(self, data_dir):
        """String representation of the classes."""
        vocab = LabelVocabulary(data_dir, narrow=False)
        decoder = LabelDecoder(data_dir)
        return [
            decoder.decode_label(vocab.id_to_label(i)) for i in range(self.num_classes)
        ]

    def hparams(self, defaults, unused_model_hparams):
        super().hparams(defaults, unused_model_hparams)
        p = defaults
        p.modality["image"] = modalities.RealModality
        p.vocab_size["image"] = _IMAGE_FEATURE_SIZE  # not used
        p.modality["price"] = modalities.IdentityModality
        p.vocab_size["price"] = 0  # not used
        p.modality["time"] = modalities.IdentityModality
        p.vocab_size["time"] = 0  # not used
        p.modality["example_id"] = modalities.IdentityModality
        p.vocab_size["example_id"] = 0  # not used
        if self.packed_length:
            raise NotImplementedError("This problem does not support packed_length")

    def example_reading_spec(self):
        data_fields, data_items_to_decoders = super().example_reading_spec()
        data_fields["image"] = tf.FixedLenFeature(
            [1, 1, _IMAGE_FEATURE_SIZE],
            tf.float32,
            default_value=np.zeros([1, 1, _IMAGE_FEATURE_SIZE]),
        )
        data_fields["price"] = tf.FixedLenFeature([1],
                                                  tf.int64,
                                                  default_value=[_DEFAULT_PRICE])
        data_fields["time"] = tf.FixedLenFeature([6],
                                                 tf.int64,
                                                 default_value=[18, 1, 1, 0, 0, 0])
        data_fields["example_id"] = tf.FixedLenFeature([], tf.string, default_value="")
        return data_fields, data_items_to_decoders

    def eval_metrics(self):
        return [metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5]

    def eval_metric_fns(self, model_hparams):
        metric_names = self.eval_metrics()
        if not all([m in metrics.METRICS_FNS for m in metric_names]):
            error_str = ("Unrecognized metric. Problem %s specified metrics "
                         "%s. Recognized metrics are %s.")
            raise ValueError(
                error_str % (self.name, metric_names, list(metrics.METRICS_FNS.keys())))
        return {
            **{
                metric_name: metrics.METRICS_FNS[metric_name]
                for metric_name in metric_names
            },
            "shopping_score": ShoppingScore(model_hparams.data_dir),
            "refined_shopping_score": RefinedShoppingScore(model_hparams.data_dir),
        }


class HierarchicalLabelDecoder:
    def __init__(self, data_dir):
        with tf.gfile.Open(os.path.join(data_dir, "category.json")) as f:
            category_ids = json.load(f)
        self.label_to_name = {
            level: dict(zip(category_ids[level].values(), category_ids[level].keys()))
            for level in CATEGORY_LEVELS
        }

    def decode_label(self, level, label):
        level_name = CATEGORY_LEVELS[level]
        return self.label_to_name[level_name].get(label, "UNK({})".format(label))


class HierarchicalLabelVocabulary:
    def __init__(self, data_dir, narrow):
        self.id_to_label, _ = load_labels(data_dir, narrow=narrow)

    def level_class_ids(self, level):
        def _fix_pad(idx):
            if idx == -1:
                return 0
            return idx

        return tf.constant([_fix_pad(label[level]) for label in self.id_to_label])


class HierarchicalShoppingScore:
    def __init__(self,
                 data_dir,
                 name="shopping_score",
                 level_weights=SCORE_WEIGHTS,
                 class_vocab_sizes=NUM_LEVEL_CLASSES):
        self.name = name
        self.vocab = HierarchicalLabelVocabulary(data_dir, narrow=True)
        self.level_weights = level_weights
        self.class_vocab_sizes = class_vocab_sizes

    def __call__(self, predictions, labels, weights_fn=common_layers.weights_nonzero):
        """
        대/중/소/세 분류별로 accuracy 를 구하고 전체 가중 점수를 계산합니다.

        Returns:
          scores: weighted score.
          weights: returns all ones.
        """

        print("** {} **".format(self.name))
        print(" - predictions: {}".format(predictions.shape))
        print(" - labels: {}".format(labels.shape))

        boundaries = [(sum(self.class_vocab_sizes[:i]),
                       sum(self.class_vocab_sizes[:(i + 1)])) for i in range(4)]
        leveled_predictions = [predictions[..., start:end] for start, end in boundaries]
        leveled_labels = [
            tf.gather(self.vocab.level_class_ids(i), labels) for i in range(4)
        ]

        level_scores = []
        level_weights = []

        if weights_fn is not common_layers.weights_nonzero:
            raise ValueError("Only weights_nonzero can be used for this metric.")

        for level in range(4):
            with tf.variable_scope(self.name, values=[predictions, labels]):
                padded_predictions, padded_labels = common_layers.pad_with_zeros(
                    leveled_predictions[level], leveled_labels[level])
                weights = weights_fn(padded_labels)
                outputs = tf.to_int32(tf.argmax(padded_predictions, axis=-1))
                padded_labels = tf.to_int32(padded_labels)
                level_scores.append(tf.to_float(tf.equal(outputs, padded_labels)))
                level_weights.append(weights * self.level_weights[level])

        final_scores = sum(
            scores * weights for scores, weights in zip(level_scores, level_weights))

        # every sample count
        weights = tf.ones(tf.shape(final_scores), dtype=tf.float32)

        return final_scores, weights


class HierarchicalShoppingScore1(HierarchicalShoppingScore):
    def __init__(self, data_dir, name="shopping_score_1"):
        super().__init__(data_dir, name, level_weights=[1, 0, 0, 0])


class HierarchicalShoppingScore2(HierarchicalShoppingScore):
    def __init__(self, data_dir, name="shopping_score_2"):
        super().__init__(data_dir, name, level_weights=[0, 1, 0, 0])


class HierarchicalShoppingScore3(HierarchicalShoppingScore):
    def __init__(self, data_dir, name="shopping_score_3"):
        super().__init__(data_dir, name, level_weights=[0, 0, 1, 0])


class HierarchicalShoppingScore4(HierarchicalShoppingScore):
    def __init__(self, data_dir, name="shopping_score_4"):
        super().__init__(data_dir, name, level_weights=[0, 0, 0, 1])


class HierarchicalShoppingClassLabelModality(modalities.ClassLabelModality):
    """Used for hiearchical shopping class labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab = HierarchicalLabelVocabulary(
            self._model_hparams.data_dir, narrow=True)

    @property
    def targets_weights_fn(self):
        """Target weight function for multi label, defaults to nonzero labels."""
        return common_layers.weights_nonzero

    def loss(self, top_out, targets, weights_fn=None):
        """
        각 레벨별로 각각 softmax 를 취합니다.
        그리고 zero label 을 loss 에서 제외합니다.
        이렇게 구한 레벨별 loss 를 weighted sum 합니다.

        Args:
          top_out: logits Tensor with shape [batch, ?, ?, num_classes]
          targets: targets Tensor with shape [batch, ?, ?, 1]
        Returns:
          loss_scale (cross-entropy), loss_denom
        """

        print("** HierarhiccalShoppingClassLabelModality: loss(top_out, targets) **")
        print(" - top_out: {}, {}".format(top_out.shape, top_out.dtype))
        print(" - targets: {}, {}".format(targets.shape, targets.dtype))

        boundaries = [(sum(NUM_LEVEL_CLASSES[:i]), sum(NUM_LEVEL_CLASSES[:(i + 1)]))
                      for i in range(4)]
        leveled_logits = [top_out[..., start:end] for start, end in boundaries]
        leveled_targets = [
            tf.gather(self.vocab.level_class_ids(i), targets) for i in range(4)
        ]

        if weights_fn is None:
            weights_fn = self.targets_weights_fn

        if weights_fn is not common_layers.weights_nonzero:
            print("*** weights_fn is not weights_nonzero ***.")

        loss_scales = []
        loss_denoms = []
        for level in range(4):
            loss_scale, loss_denom = common_layers.padded_cross_entropy(
                leveled_logits[level],
                leveled_targets[level],
                self._model_hparams.label_smoothing,
                weights_fn=weights_fn,
            )
            loss_scales.append(loss_scale)
            loss_denoms.append(loss_denom)

        final_loss_scale = sum(
            loss_scale * score_weight
            for loss_scale, score_weight in zip(loss_scales, SCORE_WEIGHTS))
        final_loss_denom = sum(
            loss_denom * score_weight
            for loss_denom, score_weight in zip(loss_denoms, SCORE_WEIGHTS))

        return final_loss_scale, final_loss_denom


@registry.register_problem
class HierarchicalShopping(Shopping):
    """
    Shopping product classification problem.

    상품 클래스는 4 단계로 구분됩니다.
    Level 1: 1 ~ 57
    Level 2: 1 ~ 552
    Level 3: 2 ~ 3190 (0 일 경우 점수에 계산되지 않습니다.)
    Level 4: 2 ~ 404 (0 일 경우 점수에 계산되지 않습니다.)

    출력 vocab 크기는 4203 (57 + 552 + 3190 + 404) 입니다.
    따라서 logits 은 4 단계로 구분된 softmax 의 입력입니다.
    """

    def hparams(self, defaults, unused_model_hparams):
        super().hparams(defaults, unused_model_hparams)
        p = defaults
        p.modality["targets"] = HierarchicalShoppingClassLabelModality
        p.vocab_size["targets"] = sum(NUM_LEVEL_CLASSES)

    def eval_metrics(self):
        return []

    def eval_metric_fns(self, model_hparams):
        return {
            "shopping_score": HierarchicalShoppingScore(model_hparams.data_dir),
            "shopping_score_b": HierarchicalShoppingScore1(model_hparams.data_dir),
            "shopping_score_m": HierarchicalShoppingScore2(model_hparams.data_dir),
            "shopping_score_s": HierarchicalShoppingScore3(model_hparams.data_dir),
            "shopping_score_d": HierarchicalShoppingScore4(model_hparams.data_dir),
        }


@registry.register_problem
class HierarchicalShoppingLogImage(HierarchicalShopping):
    def hparams(self, defaults, unused_model_hparams):
        super().hparams(defaults, unused_model_hparams)
        p = defaults
        p.modality["image"] = LogRealModality


@registry.register_problem
class ShoppingPublicLB(Shopping):
    @property
    def already_shuffled(self):
        return True

    def _task_id_to_input_file(self, tmp_dir, task_id):
        input_file = os.path.join(tmp_dir, self.EVAL_FILES[task_id])
        return input_file

    def _task_id_to_output_split(self, task_id):
        dataset_split = self.dataset_splits[0]
        assert dataset_split["split"] == problem.DatasetSplit.TEST
        augment = False
        return dataset_split, task_id, augment

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{"split": problem.DatasetSplit.TEST, "shards": len(self.EVAL_FILES)}]


@registry.register_problem
class ShoppingPrivateLB(Shopping):
    @property
    def already_shuffled(self):
        return True

    def _task_id_to_input_file(self, tmp_dir, task_id):
        input_file = os.path.join(tmp_dir, self.TEST_FILES[task_id])
        return input_file

    def _task_id_to_output_split(self, task_id):
        dataset_split = self.dataset_splits[0]
        assert dataset_split["split"] == problem.DatasetSplit.TEST
        augment = False
        return dataset_split, task_id, augment

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{"split": problem.DatasetSplit.TEST, "shards": len(self.TEST_FILES)}]


@registry.register_problem
class HierarchicalShoppingPublicLB(HierarchicalShopping):
    @property
    def already_shuffled(self):
        return True

    def _task_id_to_input_file(self, tmp_dir, task_id):
        input_file = os.path.join(tmp_dir, self.EVAL_FILES[task_id])
        return input_file

    def _task_id_to_output_split(self, task_id):
        dataset_split = self.dataset_splits[0]
        assert dataset_split["split"] == problem.DatasetSplit.TEST
        augment = False
        return dataset_split, task_id, augment

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{"split": problem.DatasetSplit.TEST, "shards": len(self.EVAL_FILES)}]


@registry.register_problem
class HierarchicalShoppingPrivateLB(HierarchicalShopping):
    @property
    def already_shuffled(self):
        return True

    def _task_id_to_input_file(self, tmp_dir, task_id):
        input_file = os.path.join(tmp_dir, self.TEST_FILES[task_id])
        return input_file

    def _task_id_to_output_split(self, task_id):
        dataset_split = self.dataset_splits[0]
        assert dataset_split["split"] == problem.DatasetSplit.TEST
        augment = False
        return dataset_split, task_id, augment

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{"split": problem.DatasetSplit.TEST, "shards": len(self.TEST_FILES)}]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


class ScoreMax:
    def __init__(self, data_dir):
        self.id_to_label, _ = load_labels(data_dir, narrow=False)
        self.score_matrix = compute_score_matrix(data_dir)

    def class_id_to_label(self, class_id):
        return self.id_to_label[class_id]

    def predict_label_by_logit(self, logits):
        """Refine label predictions

        :param logits: np.ndarray
        :returns: label
        :rtype: tuple

        """
        prob = softmax(logits)
        class_id = np.argmax(self.score_matrix @ prob)
        return self.class_id_to_label(class_id)

    def __call__(self, prediction):
        logits = prediction["logits"].ravel()
        label = self.predict_label_by_logit(logits)
        return label


class HierarchicalArgMax:
    def __init__(self, data_dir, class_vocab_sizes=NUM_LEVEL_CLASSES):
        self.class_vocab_sizes = class_vocab_sizes
        self.subclasses = load_subclasses(data_dir, narrow=True)

    def predict_label_by_logit(self, logits):
        """Refine label predictions

        :param logits: np.ndarray
        :returns: label
        :rtype: tuple

        """
        boundaries = [(sum(self.class_vocab_sizes[:i]),
                       sum(self.class_vocab_sizes[:(i + 1)])) for i in range(4)]
        leveled_logits = [logits[..., start:end] for start, end in boundaries]

        label = []
        for logits in leveled_logits:
            feasible_classes = self.subclasses.get(tuple(label), [])
            if not feasible_classes:
                feasible_classes = range(1, len(logits))
            feasible_classes = list(feasible_classes)
            idx = np.argmax(logits[feasible_classes])
            label.append(feasible_classes[idx])
        return tuple(label)

    def __call__(self, prediction):
        logits = prediction["logits"].ravel()
        label = self.predict_label_by_logit(logits)
        return label
