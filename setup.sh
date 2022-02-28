#!/bin/sh
pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl
pip install comet_ml
pip install pandas
pip install tabulate

git clone https://github.com/ray-project/ray
cd ray
git remote add upstream https://github.com/ray-project/ray.git
cp ../setup-dev.py python/ray/setup-dev.py

# This replaces `<package path>/site-packages/ray/<package>`
# with your local `ray/python/ray/<package>`.
$(which python) python/ray/setup-dev.py
