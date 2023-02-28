import platform
import tempfile
from ftplib import FTP
from pathlib import Path

from modyn.trainer_server.internal.ftp.ftp_server import FTPServer


def test_ftp_server():
    if platform.system() == "Darwin":
        # On macOS, the ftpserver works but throws a file descriptor error upon termination in tests
        return

    with tempfile.TemporaryDirectory() as tempdir:
        temppath = Path(tempdir)
        config = {"trainer_server": {"ftp_port": 1337}}
        with open(temppath / "test.test", "w") as file:  # pylint: disable=unspecified-encoding
            file.write("test")

        ftp_server = FTPServer(config, temppath)

        with ftp_server:
            ftp = FTP()
            ftp.connect("127.0.0.1", 1337, timeout=3)
            ftp.login("modyn", "modyn")

            files = []
            ftp.retrlines("MLSD", files.append)
            files = [str(file).split(" ")[1] for file in files]

            assert files == ["test.test"]
