#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
훈련된 모델의 checkpoint 에는 Adam momentum 과 variance 가 함께 저장됩니다.
이 스크립트는 실제 model weights 만 저장해서 용량을 줄입니다.
"""
import argparse
import os

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description="Predict Shopping Categories")

    ## Optional arguments
    parser.add_argument("--input_path", type=str, help="Path to the input checkpoint.")
    parser.add_argument(
        "--output_path", type=str, help="Path to the output checkpoint.")

    args = parser.parse_args()

    input_path = os.path.expanduser(args.input_path)
    output_path = os.path.expanduser(args.output_path)

    output_dir = os.path.dirname(output_path)
    tf.gfile.MakeDirs(output_dir)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.import_meta_graph(input_path + '.meta')
    saver.restore(sess, input_path)
    model_variables = [
        var for var in tf.global_variables() if not var.name.startswith('training')
    ]
    saver = tf.train.Saver(model_variables)
    saver.save(sess, output_path)


if __name__ == "__main__":
    main()
