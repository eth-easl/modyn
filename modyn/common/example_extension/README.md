# Example Extension

This directory contains an example of how to build a C++ extension for a Modyn Python component.
In `example_extension.py`, the Python wrapper around the C++ library is expoed.
It loads the DLL, and needs to provide the input/output for each function of the library.
Then, for each C++ library function, a Python function wrapper is provided.

In the `src` file, you find the `example_extension.cpp` file containing the actual implementation of the C++ library.
The `example_extension_wrapper.cpp` file uses a `extern C` statement to expose the C++ functions implemented in the main file with C function names.
This avoids C++ name wrangling.
Only the C functions can easily be loaded in Python, without having to deal with C++ symbols.
For example, here, we can load the sum function as `self._sum_list_impl = self.extension.sum_list` instead of `self._sum_list_impl = self.extension.__ZN5modyn6common17example_extension13sum_list_implEPKyy`.

The `CMakeLists.txt` file adds the `example_extension` shared library to the project.
There are both C++ unit tests and Python unit tests, making sure the library and the Python wrapper work, in `modyn/tests/common/example_extension`.
