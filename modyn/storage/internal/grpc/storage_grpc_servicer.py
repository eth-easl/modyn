import grpc
import logging
import typing
from typing import List

from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageServicer
# pylint: disable-next=no-name-in-module
from modyn.storage.internal.grpc.generated.storage_pb2 import GetRequest, GetResponse, \
    GetNewDataSinceRequest, GetNewDataSinceResponse, DatasetAvailableRequest, \
    DatasetAvailableResponse, RegisterNewDatasetRequest, RegisterNewDatasetResponse, \
    GetDataInIntervalRequest, GetDataInIntervalResponse
from modyn.storage.internal.database.database_connection import DatabaseConnection
from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.database.models.file import File
from modyn.storage.internal.database.models.sample import Sample
from modyn.storage.internal.database.storage_database_utils import get_file_wrapper

logger = logging.getLogger(__name__)


class StorageGRPCServicer(StorageServicer):
    def __init__(self, config: dict):
        self.modyn_config = config
        self.database = DatabaseConnection(config)
        super().__init__()

    # pylint: disable-next=unused-argument,invalid-name
    def Get(self, request: GetRequest, context: grpc.ServicerContext) -> typing.Iterable[GetResponse]:
        with DatabaseConnection(self.modyn_config) as database:
            session = database.get_session()

            dataset: Dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()
            if dataset is None:
                logger.error(f'Dataset with name {request.dataset_id} does not exist.')
                yield GetResponse()
                return

            samples: typing.List[Sample] = session.query(Sample) \
                .filter(Sample.external_key.in_(request.keys)).order_by(Sample.file_id).all()

            if len(samples) == 0:
                logger.error('No samples found in the database.')
                yield GetResponse()
                return

            if len(samples) != len(request.keys):
                logger.error('Not all keys were found in the database.')
                not_found_keys = {s for s in request.keys if s not in [sample.external_key for sample in samples]}
                logger.error(f'Keys: {not_found_keys}')

            file_id = samples[0].file_id
            samples_per_file: List[int] = []

            for sample in samples:
                if sample.file_id != file_id:
                    file_id = sample.file_id
                    file: File = sample.file
                    file_wrapper = get_file_wrapper(dataset.file_wrapper_type, file.path)
                    yield GetResponse(chunk=file_wrapper.get_samples_from_indices(samples_per_file))
                else:
                    samples_per_file.append(sample.index)

    # pylint: disable-next=unused-argument,invalid-name
    def GetNewDataSince(self, request: GetNewDataSinceRequest, context: grpc.ServicerContext)\
            -> GetNewDataSinceResponse:
        with DatabaseConnection(self.modyn_config) as database:
            session = database.get_session()

            dataset: Dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()

            if dataset is None:
                logger.error(f'Dataset with name {request.dataset_id} does not exist.')
                return GetNewDataSinceResponse()

            timestamp = request.timestamp

            external_keys = session.query(Sample.external_key) \
                .join(File) \
                .filter(File.dataset_id == dataset.id) \
                .filter(File.updated_at >= timestamp) \
                .all()

            if len(external_keys) == 0:
                logger.info(f'No new data since {timestamp}')
                return GetNewDataSinceResponse()

            return GetNewDataSinceResponse(keys=[external_key[0] for external_key in external_keys])

    def GetDataInInterval(self, request: GetDataInIntervalRequest, context: grpc.ServicerContext)\
            -> GetDataInIntervalResponse:
        with DatabaseConnection(self.modyn_config) as database:
            session = database.get_session()

            dataset: Dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()

            if dataset is None:
                logger.error(f'Dataset with name {request.dataset_id} does not exist.')
                return GetDataInIntervalResponse()

            external_keys = session.query(Sample.external_key) \
                .join(File) \
                .filter(File.dataset_id == dataset.id) \
                .filter(File.updated_at >= request.start_timestamp) \
                .filter(File.updated_at <= request.end_timestamp) \
                .all()

            if len(external_keys) == 0:
                logger.info(f'No data between timestamp {request.start_timestamp} and {request.end_timestamp}')
                return GetDataInIntervalResponse()

            return GetDataInIntervalResponse(keys=[external_key[0] for external_key in external_keys])

    # pylint: disable-next=unused-argument,invalid-name
    def CheckAvailability(self, request: DatasetAvailableRequest, context: grpc.ServicerContext) \
            -> DatasetAvailableResponse:
        with DatabaseConnection(self.modyn_config) as database:
            session = database.get_session()

            dataset: Dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()

            if dataset is None:
                logger.error(f'Dataset with name {request.dataset_id} does not exist.')
                return DatasetAvailableResponse(available=False)

            return DatasetAvailableResponse(available=True)

    # pylint: disable-next=unused-argument,invalid-name
    def RegisterNewDataset(self, request: RegisterNewDatasetRequest, context: grpc.ServicerContext)\
            -> RegisterNewDatasetResponse:
        with DatabaseConnection(self.modyn_config) as database:
            success = database.add_dataset(request.dataset_id,
                                           request.base_path,
                                           request.filesystem_wrapper_type,
                                           request.file_wrapper_type,
                                           request.description,
                                           request.version)
            return RegisterNewDatasetResponse(success=success)
