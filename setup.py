#!/usr/bin/env python

import sys
from setuptools import setup, find_packages
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [2, 0], "Requires PyTorch >= 2.0"

WHEEL_URLS = {(3,10): "https://github.com/agoryuno/adet_layers/raw/main/wheels/adet_layers-0.0.1-cp310-cp310-linux_x86_64.whl",
              (3,11): "https://github.com/agoryuno/adet_layers/raw/main/wheels/adet_layers-0.0.1-cp311-cp311-linux_x86_64.whl"}

def get_wheel(wheels=WHEEL_URLS):
    major = sys.version_info.major
    minor = sys.version_info.minor

    wheel_url = wheels.get((major, minor))
    assert wheel_url, f"Couldn't locate a .whl file of the 'adet_layers' library for Python version {major}.{minor}"
    return wheel_url


setup(
    name="deepsolo_onnx",
    version="0.0.1",
    author="Alex Goryunov",
    url="https://github.com/agoryuno/deepsolo_onnx",
    description="A stripped down version of the original "
        " DeepSolo model's codebase, with most dependencies removed or "
        " repackaged separately.",
    packages=find_packages(exclude=["adet", "adet.*"]),
    python_requires=">=3.10, <3.12",
    install_requires=[
        f"adet_layers @ {get_wheel()}",
        "dconfig @ git+https://github.com/agoryuno/dconfig"
    ],
    extras_require={"all": ["psutil"]}
)
