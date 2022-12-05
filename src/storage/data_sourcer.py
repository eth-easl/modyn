import os
import sys
import time 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def serve(config: dict):
    if (config['storage']['data_source']['enabled']):
        source_module = my_import('storage.datasource')
        source = getattr(
            source_module,
            config['storage']['data_source']['type'])(config)
        source.run()

def my_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

if __name__ == '__main__':
    import sys
    import yaml
    import logging

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python storage_server.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    # Wait for the database to be ready
    time.sleep(10)
    serve(config)