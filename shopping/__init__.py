# pylint: disable=wrong-import-position

# Make sure that shopping is running on Python 3.6.0 or later
import sys

if sys.version_info < (3, 6, 0):
    raise RuntimeError("Shopping requires Python 3.6.0 or later")


import shopping.data.datasets
import shopping.models
