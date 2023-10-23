import ctypes
from pathlib import Path
from sys import platform


class ExampleExtension:
    @staticmethod
    def __get_library_path():
        if platform == "darwin":
            library_filename = "libexample_extension.dylib"
        else:
            library_filename = "libexample_extension.so"

        return ExampleExtension.__get_build_path() / library_filename

    @staticmethod
    def __get_build_path():
        return Path(__file__).parent.parent.parent.parent / "libbuild"

    @staticmethod
    def __ensure_library_present():
        path = ExampleExtension.__get_library_path()
        if not path.exists():
            raise RuntimeError(f"Cannot find {ExampleExtension.__name__} library at {path}")

    def __init__(self) -> None:
        ExampleExtension.__ensure_library_present()
        self.extension = ctypes.CDLL(ExampleExtension.__get_library_path())

        self._sum_list_impl = self.extension.sum_list
        self._sum_list_impl.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64]
        self._sum_list_impl.restype = ctypes.c_uint64

    def sum_list(self, data_list: list[int]) -> int:
        IntArray = ctypes.c_uint64 * len(data_list)
        result = self._sum_list_impl(IntArray(*data_list), len(data_list))

        return result
