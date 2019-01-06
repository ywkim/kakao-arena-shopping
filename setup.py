#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from setuptools import find_packages, setup

# Package meta-data.
NAME = "shopping"
DESCRIPTION = "Shopping"
URL = "https://github.com/ywkim/kakao-arena-shopping"
EMAIL = "youngwook.kim@gmail.com"
AUTHOR = "Youngwook Kim"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = "0.1.0"

# What packages are required for this module to be executed?
REQUIRED = ["tensor2tensor>=1.11.0,<2.0.0", "sentencepiece", "tqdm"]

# What packages are optional?
EXTRAS = {
    "tensorflow": ["tensorflow>=1.12.0,<2.0.0"],
    "tensorflow_gpu": ["tensorflow-gpu>=1.12.0,<2.0.0"]
}

here = os.path.abspath(os.path.dirname(__file__))

# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
