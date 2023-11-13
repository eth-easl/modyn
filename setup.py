"""setup file for the project."""
# code inspired by https://github.com/navdeep-G/setup.py

import io
import os
import pathlib
import subprocess
from pprint import pprint

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# Package meta-data.
NAME = "modyn"
DESCRIPTION = "A platform for training on dynamic datasets."

URL = "https://github.com/eth-easl/dynamic_datasets_dsl"
URL_DOKU = "https://github.com/eth-easl/dynamic_datasets_dsl"
URL_GITHUB = "https://github.com/eth-easl/dynamic_datasets_dsl"
URL_ISSUES = "https://github.com/eth-easl/dynamic_datasets_dsl/issues"
EMAIL = "maximilian.boether@inf.ethz.ch"
AUTHOR = "See contributing.md"
REQUIRES_PYTHON = ">=3.9"
KEYWORDS = [""]
# TODO: What packages are required for this module to be executed?
REQUIRED = [""]


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
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's _version.py module as a dictionary.
about = {}
project_slug = "modyn"

EXTENSION_BUILD_DIR = pathlib.Path(here) / "libbuild"


def _get_env_variable(name, default="OFF"):
    if name not in os.environ.keys():
        return default
    return os.environ[name]


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir=".", sources=[], **kwa):
        Extension.__init__(self, name, sources=sources, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class CMakeBuild(build_ext):
    def copy_extensions_to_source(self):
        pass

    def build_extensions(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("Cannot find CMake executable")

        for ext in self.extensions:
            cfg = _get_env_variable("MODYN_BUILDTYPE", "Release")
            print(f"Using build type {cfg} for Modyn.")
            cmake_args = [
                "-DCMAKE_BUILD_TYPE=%s" % cfg,
                "-DMODYN_BUILD_PLAYGROUND=Off",
                "-DMODYN_BUILD_TESTS=Off",
                "-DMODYN_BUILD_STORAGE=Off",
                "-DMODYN_TEST_COVERAGE=Off",
            ]

            pprint(cmake_args)

            if not os.path.exists(EXTENSION_BUILD_DIR):
                os.makedirs(EXTENSION_BUILD_DIR)

            # Config and build the extension
            subprocess.check_call(["cmake", ext.cmake_lists_dir] + cmake_args, cwd=EXTENSION_BUILD_DIR)
            subprocess.check_call(["cmake", "--build", ".", "--config", cfg], cwd=EXTENSION_BUILD_DIR)


# Where the magic happens:
setup(
    name=NAME,
    version="1.0.0",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    project_urls={"Bug Tracker": URL_ISSUES, "Source Code": URL_GITHUB},
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*", "tests.*.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # entry_points is is required for testing the Python scripts
    entry_points={
        "console_scripts": [
            "_modyn_supervisor=modyn.supervisor.entrypoint:main",
            "_modyn_storage=modyn.storage.storage_entrypoint:main",
            "_modyn_trainer_server=modyn.trainer_server.trainer_server_entrypoint:main",
            "_modyn_selector=modyn.selector.selector_entrypoint:main",
            "_modyn_metadata_processor=modyn.metadata_processor.metadata_processor_entrypoint:main",
            "_modyn_model_storage=modyn.model_storage.model_storage_entrypoint:main",
            "_modyn_evaluator=modyn.evaluator.evaluator_entrypoint:main",
        ]
    },
    scripts=[
        "modyn/supervisor/modyn-supervisor",
        "modyn/storage/modyn-storage",
        "modyn/trainer_server/modyn-trainer-server",
        "modyn/selector/modyn-selector",
        "modyn/metadata_processor/modyn-metadata-processor",
        "modyn/model_storage/modyn-model-storage",
        "modyn/evaluator/modyn-evaluator",
    ],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    keywords=KEYWORDS,
    ext_modules=[CMakeExtension("example_extension"), CMakeExtension("trigger_sample_storage")],
    cmdclass={"build_ext": CMakeBuild},
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
