import grpc
import logging
import typing
import datetime

from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageServicer
# pylint: disable-next=no-name-in-module
from modyn.storage.internal.grpc.generated.storage_pb2 import GetRequest, GetResponse, \
    GetNewDataSinceRequest, GetNewDataSinceResponse, DatasetAvailableRequest, \
    DatasetAvailableResponse, RegisterNewDatasetRequest, RegisterNewDatasetResponse
from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.database.models.file import File
from modyn.storage.internal.database.models.sample import Sample
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import InvalidFilesystemWrapperTypeException
from modyn.storage.internal.file_wrapper.file_wrapper_type import InvalidFileWrapperTypeException
from modyn.storage.internal.database.storage_database_utils import get_filesystem_wrapper, get_file_wrapper

logger = logging.getLogger(__name__)


class StorageGRPCServer(StorageServicer):
    def __init__(self, config: dict):
        self.modyn_config = config
        self.database = DatabaseConnection(config)
        super().__init__()

    # pylint: disable-next=unused-argument,invalid-name
    def Get(self, request: GetRequest, context: grpc.ServicerContext) -> typing.Iterable[GetResponse]:
        session = self.database.get_session()

        dataset: Dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()
        if dataset is None:
            logger.error(f'Dataset with name {request.dataset_id} does not exist.')
            yield GetResponse()

        try:
            filesystem_wrapper = get_filesystem_wrapper(dataset.filesystem_wrapper_type, dataset.base_path)
        except InvalidFilesystemWrapperTypeException as exception:
            logger.error(f'Invalid filesystem wrapper type: {exception}')
            yield GetResponse()

        samples: typing.List[Sample] = session.query(Sample) \
            .filter(Sample.external_key.in_(request.keys)).all()

        if samples is None:
            logger.error('No samples found in the database.')
            yield GetResponse()

        if len(samples) != len(request.keys):
            logger.error('Not all keys were found in the database.')
            not_found_keys = {s for s in request.keys if s not in [sample.external_key for sample in samples]}
            logger.error(f'Keys: {not_found_keys}')

        #  TODO(vGsteiger): Check if all samples are from the same file.
        #  TODO(vGsteiger): Optimize by reading the file only once.
        #  TODO(vGsteiger): Optimize by caching the most used samples.
        for sample in samples:
            file: File = sample.file
            try:
                file_wrapper = get_file_wrapper(filesystem_wrapper.filesystem_wrapper_type, file.path)
            except InvalidFileWrapperTypeException as exception:
                logger.error(f'Invalid file wrapper type: {exception}')
                yield GetResponse()
            yield GetResponse(chunk=file_wrapper.get_sample(sample.index))

    # pylint: disable-next=unused-argument,invalid-name
    def GetNewDataSince(self, request: GetNewDataSinceRequest, context: grpc.ServicerContext)\
            -> GetNewDataSinceResponse:
        session = self.database.get_session()

        dataset: Dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()

        if dataset is None:
            logger.error(f'Dataset with name {request.dataset_id} does not exist.')
            return GetNewDataSinceResponse()

        timestamp = datetime.datetime.fromtimestamp(request.timestamp)

        external_keys = session.query(Sample.external_key) \
            .filter(Sample.file_id.in_(dataset.files)) \
            .filter(Sample.timestamp > timestamp) \
            .all()

        if len(external_keys) == 0:
            logger.info(f'No new data since {timestamp}')
            return GetNewDataSinceResponse()

        return GetNewDataSinceResponse(keys=[external_key[0] for external_key in external_keys])

    # pylint: disable-next=unused-argument,invalid-name
    def CheckAvailability(self, request: DatasetAvailableRequest, context: grpc.ServicerContext) \
            -> DatasetAvailableResponse:
        session = self.database.get_session()

        dataset: Dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()

        if dataset is None:
            logger.error(f'Dataset with name {request.dataset_id} does not exist.')
            return DatasetAvailableResponse(available=False)

        return DatasetAvailableResponse(available=True)

    # pylint: disable-next=unused-argument,invalid-name
    def RegisterNewDataset(self, request: RegisterNewDatasetRequest, context: grpc.ServicerContext)\
            -> RegisterNewDatasetResponse:
        success = self.database.add_dataset(request.dataset_id,
                                            request.base_path,
                                            request.file_wrapper_type,
                                            request.file_wrapper_type,
                                            request.description)
        return RegisterNewDatasetResponse(success=success)
