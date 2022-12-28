import grpc
import logging
import typing
import datetime

from modyn.storage.internal.grpc.storage_pb2_grpc import StorageServicer
from modyn.storage.internal.grpc.storage_pb2 import GetRequest, GetResponse, GetNewDataSinceRequest, GetNewDataSinceResponse, DatasetAvailableRequest, DatasetAvailableResponse, RegisterNewDatasetRequest, RegisterNewDatasetResponse
from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.database.models.file import File
from modyn.storage.internal.database.models.sample import Sample
from modyn.storage.internal.database.utils import get_file_system_wrapper, get_file_wrapper

logger = logging.getLogger(__name__)


class StorageGRPCServer(StorageServicer):
    def __init__(self, config: dict):
        self.modyn_config = config
        self.database = DatabaseConnection(config)
        super().__init__()

    def Get(self, request: GetRequest, context: grpc.ServicerContext) -> GetResponse:
        session = self.database.get_session()

        dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()
        if dataset is None:
            logger.error('Dataset with name %s does not exist.', request.dataset_id)
            return GetResponse()

        file_system_wrapper = get_file_system_wrapper(dataset.file_system_wrapper_type, dataset.base_path)

        samples: typing.List[Sample] = session.query(Sample).filter(Sample.external_key.in_(request.keys)).all()

        if len(samples) != len(request.keys):
            logger.error('Not all keys were found in the database.')
            not_found_keys = set(request.keys) - set([sample.external_key for sample in samples])
            logger.error('Keys: %s', not_found_keys)

        # TODO: Check if all samples are from the same file.
        # TODO: Optimize by reading the file only once.
        # TODO: Optimize by caching the most used samples.
        for sample in samples:
            file: File = sample.file
            file_wrapper = get_file_wrapper(file_system_wrapper.filesystem_wrapper_type, file.path)
            yield GetResponse(chunk=file_wrapper.get_sample(sample.index))

    def GetNewDataSince(self, request: GetNewDataSinceRequest, context: grpc.ServicerContext) -> GetNewDataSinceResponse:
        session = self.database.get_session()

        dataset: Dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()

        if dataset is None:
            logger.error('Dataset with name %s does not exist.', request.dataset_id)
            return GetNewDataSinceResponse()

        timestamp = datetime.datetime.fromtimestamp(request.timestamp)

        external_keys = session.query(Sample.external_key).filter(Sample.file_id.in_(dataset.files)).filter(Sample.timestamp > timestamp).all()

        if len(external_keys) == 0:
            logger.info('No new data since %s', timestamp)
            return GetNewDataSinceResponse()

        return GetNewDataSinceResponse(external_keys=[external_key[0] for external_key in external_keys])

    def CheckAvailability(self, request: DatasetAvailableRequest, context: grpc.ServicerContext) -> DatasetAvailableResponse:
        session = self.database.get_session()

        dataset: Dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()

        if dataset is None:
            logger.error('Dataset with name %s does not exist.', request.dataset_id)
            return DatasetAvailableResponse(available=False)
        
        return DatasetAvailableResponse(available=True)

    def RegisterNewDataset(self, request: RegisterNewDatasetRequest, context: grpc.ServicerContext) -> RegisterNewDatasetResponse:
        success = self.database.add_dataset(request.dataset_id, request.base_path, request.filesystem_type, request.file_type, request.description)
        return RegisterNewDatasetResponse(success=success)
