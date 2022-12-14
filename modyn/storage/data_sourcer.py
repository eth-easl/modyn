import sys
import time
import logging

import yaml

from config import dynamic_module_import


def serve(config: dict):
    if (config['storage']['data_source']['enabled']):
        source_module = dynamic_module_import('storage.datasource')
        source = getattr(
            source_module,
            config['storage']['data_source']['type'])(config)
        source.run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python storage_server.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    # Wait for the database to be ready
    time.sleep(10)
    serve(config)
