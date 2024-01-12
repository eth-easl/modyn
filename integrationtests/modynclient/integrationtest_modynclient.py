import subprocess

from integrationtests.utils import CLIENT_CONFIG_FILE, CLIENT_ENTRYPOINT, DUMMY_CONFIG_FILE, TinyDatasetHelper

if __name__ == "__main__":
    dataset_helper = TinyDatasetHelper()
    try:
        dataset_helper.setup_dataset()
        rc = subprocess.call(
            [
                CLIENT_ENTRYPOINT,
                DUMMY_CONFIG_FILE,
                CLIENT_CONFIG_FILE,
                ".",
                "--start-replay-at",
                "0",
                "--maximum-triggers",
                "2",
            ]
        )
        assert rc == 0
    finally:
        dataset_helper.cleanup_dataset_dir()
        dataset_helper.cleanup_storage_database()
