import logging
from threading import Thread
import os

from modyn.utils import validate_yaml
from modyn.storage.internal.grpc.grpc_server import GRPCServer
from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.seeker import Seeker

logger = logging.getLogger(__name__)


class Storage():
    def __init__(self, modyn_config: dict) -> None:
        self.modyn_config = modyn_config

        if not self.validate_config():
            raise ValueError("Invalid system configuration")

    def validate_config(self) -> bool:
        schema_path = os.path.join(os.path.dirname(__file__), 'config', 'schema', 'modyn_config_schema.yaml')
        return validate_yaml(self.modyn_config, schema_path)

    def run(self) -> None:

        if not validate_yaml(self.modyn_config,
                             os.path.join(os.path.dirname(__file__),
                                          'config',
                                          'schema',
                                          'modyn_config_schema.yaml')):
            logger.error('Invalid modyn config.')
            return

        #  Create the database tables.
        with DatabaseConnection(self.modyn_config) as database:
            database.create_all()

            for dataset in self.modyn_config['storage']['datasets']:
                database.add_dataset(dataset['name'],
                                     dataset['base_path'],
                                     dataset['filesystem_wrapper_type'],
                                     dataset['file_wrapper_type'],
                                     '')

        #  Start the seeker process in a different thread.
        seeker = Thread(target=Seeker(self.modyn_config).run)

        #  Start the storage grpc server.
        with GRPCServer(self.modyn_config) as server:
            server.wait_for_termination()

        seeker.join()
