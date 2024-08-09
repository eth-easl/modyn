# DeepCore Utils

The content of this folder is taken from DeepCore with minor fixes.
Compared to DeepCore, the `else` branch of `orthogonal_matching_pursuit` (line 47) has been changed to support the updated version
of `torch.linalg.lstsq`. Refer to this [issue](https://github.com/PatrickZH/DeepCore/issues/10) for the fix.

You can find the original code [here](https://github.com/PatrickZH/DeepCore/tree/main/deepcore/methods/methods_utils) and
the MIT license [here](https://raw.githubusercontent.com/PatrickZH/DeepCore/main/LICENSE.md)

## DEEPCORE license

MIT License

Copyright (c) 2023 ZHAO, BO

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
