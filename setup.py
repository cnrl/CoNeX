#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["numpy>=1.23.4", "torch>=1.13.1", "pymonntorch>=0.1.0"]

test_requirements = [
    "pytest>=3",
]

setup(
    author="CNRL",
    author_email="ashenatena@gmail.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    description="Cortical Network for everything!",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="CNRL-CoNeX",
    name="CNRL-CoNeX",
    packages=["conex"],
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/cnrl/CoNeX",
    version="0.1.2",
    zip_safe=False,
)
