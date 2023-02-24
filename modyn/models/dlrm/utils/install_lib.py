import os
from pathlib import Path

import modyn


def install_cuda_extensions_if_not_present() -> None:
    cwd = os.getcwd()
    modyn_base_path = Path(modyn.__path__[0])
    dlrm_path = modyn_base_path / "models/dlrm"
    dlrm_cuda_ext_path = dlrm_path / "cuda_ext"
    shared_libraries = list(dlrm_cuda_ext_path.glob("*.so"))

    if len(shared_libraries) < 4:
        # install
        os.system(f"cd {dlrm_path} && pip install -v -e . && cd {cwd}")
