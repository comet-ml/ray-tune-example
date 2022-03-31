#!/bin/sh
pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl
pip install comet_ml
pip install pandas
pip install tabulate
pip install "ray[tune]"