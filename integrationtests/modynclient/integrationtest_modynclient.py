import subprocess

from integrationtests.utils import CLIENT_CONFIG_FILE, CLIENT_ENTRYPOINT, MNIST_CONFIG_FILE

if __name__ == "__main__":
    rc = subprocess.call(
        [
            CLIENT_ENTRYPOINT,
            MNIST_CONFIG_FILE,
            CLIENT_CONFIG_FILE,
            ".",
            "--start-replay-at",
            "0",
            "--maximum-triggers",
            "2",
        ]
    )
