import multiprocessing
from setuptools import setup, find_packages
import os
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "ElephantCallsAI",
    version = "0.1",
    packages = find_packages(),

    # Dependencies on other packages:
    # Couldn't get numpy install to work without
    # an out-of-band: sudo apt-get install python-dev
    setup_requires   = ['pytest-runner'],
    install_requires = ['requests>=2.21.0',
                        'matplotlib>=3.3.0',
                        'sklearn',
                        'PTable>=0.9.2',
                        'torch>=1.6.0',
                        'torchaudio>=0.6.0',
                        'torchvision>=0.7.0',
                        'pandas>=1.1.0',
                        'numpy>=1.19.1',
                        'scipy>=1.5.2',
                        'soundfile',
                        'tensorboard>=2.3.0',
                        'GPUtil>=1.4.0',
                        'logging-singleton>=0.0.3',
                        ],

    #dependency_links = ['https://github.com/DmitryUlyanov/Multicore-TSNE/tarball/master#egg=package-1.0']
    # Unit tests; they are initiated via 'python setup.py test'
    #test_suite       = 'nose.collector',
    #test_suite       = 'tests',
    #test_suite        = 'unittest2.collector',
    tests_require    =['pytest',
                       'testfixtures>=6.14.1',
                       ],

    # metadata for upload to PyPI
    author = "Nikita-Girey Nechvet Demir, Jonathan Michael Gomes Selman, Andreas Paepcke",
    author_email = "paepcke@cs.stanford.edu",
    description = "Detecting elephant calls",
    long_description_content_type = "text/markdown",
    long_description = long_description,
    license = "BSD",
    keywords = "elephants",
    url = "https://github.com/N-Demir/ElephantCallAI.git",   # project home page, if any
)
