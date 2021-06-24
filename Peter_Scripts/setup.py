import multiprocessing
from setuptools import setup, find_packages
import os
import glob

with open("README", "r") as fh:
    long_description = fh.read()

setup(
    name = "ElephantCallsAI-Peter_Scripts",
    version = "0.1",
    packages = find_packages(),

    # Dependencies on other packages:
    # Couldn't get numpy install to work without
    # an out-of-band: sudo apt-get install python-dev
    setup_requires   = ['pytest-runner'],
    install_requires = ['requests>=2.21.0',
                        'matplotlib>=3.1.3',
                        'sklearn',
                        'PTable>=0.9.2',
                        'torch>=1.5.1',
                        'torchaudio>=0.5.1',
                        'pandas>=1.0.5',
                        'torchvision>=0.7.0',
                        'tensorboardX'
                        ],

    # metadata for upload to PyPI
    author = "Nikita-Girey Nechvet Demir, Jonathan Michael Gomes Selman, Andreas Paepcke",
    author_email = "jgs8@cs.stanford.edu",
    description = "Detecting elephant calls",
    long_description_content_type = "text/markdown",
    long_description = long_description,
    license = "BSD",
    keywords = "elephants",
    url = "https://github.com/N-Demir/ElephantCallAI.git",   # project home page, if any
)
