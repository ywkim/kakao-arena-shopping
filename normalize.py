#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
특수 기호를 삭제하고, 소문자로 변환합니다.
"""

import argparse
import os

from tqdm.auto import tqdm

from shopping.data.datasets.shopping import normalize


def main():
    parser = argparse.ArgumentParser(description="Convert h5 to TSV")

    parser.add_argument("--input_file", type=str, help="Input file name.")
    parser.add_argument("--output_file", type=str, help="Output file name.")

    args = parser.parse_args()

    input_filepath = os.path.expanduser(args.input_file)
    output_filepath = os.path.expanduser(args.output_file)

    with open(output_filepath, "w", encoding="utf-8") as output_file:
        with open(input_filepath, encoding="utf-8") as input_file:
            for line in tqdm(input_file):
                normalized = normalize(line.rstrip())
                output_file.write(normalized + "\n")


if __name__ == "__main__":
    main()
