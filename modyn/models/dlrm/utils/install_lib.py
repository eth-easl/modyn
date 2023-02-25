import os
from pathlib import Path
import subprocess

import modyn


def install_cuda_extensions_if_not_present() -> None:
    cwd = os.getcwd()
    modyn_base_path = Path(modyn.__path__[0])
    dlrm_path = modyn_base_path / "models" / "dlrm"
    dlrm_cuda_ext_path = dlrm_path / "cuda_ext"
    shared_libraries = sorted(list(dlrm_cuda_ext_path.glob("*.so")))
    shared_libraries_names = [lib.split(".")[0] for lib in shared_libraries]

    if shared_libraries_names != [ "fused_embedding", "interaction_volta", "interaction_ampere", "sparse_gather"]:
        # install
        subprocess.run(["pip", "install", "-v", "-e", "."], check=True, cwd=cwd)
