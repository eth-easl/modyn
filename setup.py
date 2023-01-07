"""setup file for the project."""
# code inspired by https://github.com/navdeep-G/setup.py

import io
import os

from setuptools import find_packages, setup

# Package meta-data.
NAME = 'modyn'
DESCRIPTION = \
    'A platform for training on dynamic datasets.'

URL = 'https://github.com/eth-easl/dynamic_datasets_dsl'
URL_DOKU = "https://github.com/eth-easl/dynamic_datasets_dsl"
URL_GITHUB = "https://github.com/eth-easl/dynamic_datasets_dsl"
URL_ISSUES = "https://github.com/eth-easl/dynamic_datasets_dsl/issues"
EMAIL = 'maximilian.boether@inf.ethz.ch'
AUTHOR = 'See contributing.md'
REQUIRES_PYTHON = '>=3.9'
KEYWORDS = [""]
# TODO: What packages are required for this module to be executed?
REQUIRED = ['']


# What packages are optional?
# 'fancy feature': ['django'],}
EXTRAS = {}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's _version.py module as a dictionary.
about = {}
project_slug = "modyn"


# Where the magic happens:
setup(
    name=NAME,
    version="1.0.0",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    project_urls={
        "Bug Tracker": URL_ISSUES,
        "Source Code": URL_GITHUB,
    },
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*", "tests.*.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points is is required for testing the Python scripts
    entry_points={'console_scripts':
                  ["_modyn_supervisor=modyn.backend.supervisor.entrypoint:main",
                   "_modyn_storage=modyn.storage.storage_entrypoint:main"]},
    scripts=['modyn/backend/supervisor/modyn-supervisor', 'modyn/storage/modyn-storage'],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    keywords=KEYWORDS,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
