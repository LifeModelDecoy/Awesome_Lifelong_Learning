# -*- coding: utf-8 -*-
# @Author  : ChengShuo
# @Email   : 2021200940@ruc.edu.cn
# @Time    : 2023/5/31 20:25
import setuptools
from dgc.version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lifelong_learning",
    version="1.0.0",
    author="ChengShuo",
    author_email="2021200940@ruc.edu.cn",
    description="Awesome Lifelong Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LifeModelDecoy/Awesome_Lifelong_Learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'munkres',
        'scikit_learn',
        'tqdm',
    ],
    py_requires=["dgc"],
    python_requires='>=3.6',
)
