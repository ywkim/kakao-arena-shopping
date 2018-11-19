#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
h5 형식으로 저장된 텍스트 데이터를 TSV 형식으로 변환합니다.
"""

import argparse
import os

from tqdm.auto import tqdm

from shopping.data.datasets.shopping import DatasetReader, FeatureKeys


def main():
    parser = argparse.ArgumentParser(description="Convert h5 to TSV")

    ## Optional arguments
    parser.add_argument("--input_file", type=str, help="H5 file name.")

    parser.add_argument("--output_file", default=None, type=str, help="TSV file name.")

    args = parser.parse_args()

    input_file = os.path.expanduser(args.input_file)
    if args.output_file:
        output_file = os.path.expanduser(args.output_file)
    else:
        output_file = input_file + ".tsv"

    ds = DatasetReader([input_file])

    iterators = [
        ds.text_column_generator(FeatureKeys.ID),
        ds.text_column_generator(FeatureKeys.TITLE),
        ds.text_column_generator(FeatureKeys.MODEL),
        ds.text_column_generator("brand"),
        ds.text_column_generator("maker"),
        ds.column_generator("price"),
        ds.text_column_generator("updttm"),
        ds.column_generator("bcateid"),
        ds.column_generator("mcateid"),
        ds.column_generator("scateid"),
        ds.column_generator("dcateid"),
    ]

    def remove_tabs(s):
        return " ".join(s.split())

    with open(output_file, "w", encoding="utf-8") as fp:
        for features in tqdm(zip(*iterators), total=ds.size()):
            features = map(str, features)
            features = map(remove_tabs, features)
            fp.write("\t".join(features) + "\n")
            fp.flush()


if __name__ == "__main__":
    main()
