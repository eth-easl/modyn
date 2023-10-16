# CLOC Data

In this directory, you can find the files necessary to run experiments with the CLOC dataset. 
The dataset was introduced [in this paper](https://arxiv.org/pdf/2108.09020.pdf), and we make use of the [CLDatasets](https://github.com/hammoudhasan/CLDatasets) mirror.

## Data Generation
To run the downloading script you need to install the `google-cloud-storage` package. 
Then, you can use the `data_generation.py` script to download the data and set the timestamps accordingly. 
Use the `-h` flag to find out more.

## License

The CLOC Dataset comes with the MIT License.
The CLDatasets repository does not have an explicit license but allows to "adapt the code to suit your specific requirements".

### CLOC License Copy

MIT License

Copyright (c) 2021 Intel Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.