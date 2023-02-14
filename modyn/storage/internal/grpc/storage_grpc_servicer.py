"""Storage GRPC servicer."""

import logging
from typing import Iterable, Tuple

import grpc
from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.database.models.file import File
from modyn.storage.internal.database.models.sample import Sample
from modyn.storage.internal.database.storage_database_connection import StorageDatabaseConnection
from modyn.storage.internal.database.storage_database_utils import get_file_wrapper, get_filesystem_wrapper

# pylint: disable-next=no-name-in-module
from modyn.storage.internal.grpc.generated.storage_pb2 import (
    DatasetAvailableRequest,
    DatasetAvailableResponse,
    DeleteDatasetResponse,
    GetCurrentTimestampResponse,
    GetDataInIntervalRequest,
    GetDataInIntervalResponse,
    GetNewDataSinceRequest,
    GetNewDataSinceResponse,
    GetRequest,
    GetResponse,
    RegisterNewDatasetRequest,
    RegisterNewDatasetResponse,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageServicer
from modyn.utils.utils import current_time_millis

logger = logging.getLogger(__name__)


class StorageGRPCServicer(StorageServicer):
    """GRPC servicer for the storage module."""

    def __init__(self, config: dict):
        """Initialize the storage GRPC servicer.

        Args:
            config (dict): Configuration of the storage module.
        """
        self.modyn_config = config
        super().__init__()

    # pylint: disable-next=unused-argument,invalid-name
    def Get(self, request: GetRequest, context: grpc.ServicerContext) -> Iterable[GetResponse]:
        """Return the data for the given keys.

        Args:
            request (GetRequest): Request containing the dataset name and the keys.
            context (grpc.ServicerContext): Context of the request.

        Returns:
            Iterable[GetResponse]: Response containing the data for the given keys.

        Yields:
            Iterator[Iterable[GetResponse]]: Response containing the data for the given keys.
        """
        with StorageDatabaseConnection(self.modyn_config) as database:
            session = database.session

            dataset: Dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()
            if dataset is None:
                logger.error(f"Dataset with name {request.dataset_id} does not exist.")
                yield GetResponse()
                return

            samples: list[Sample] = (
                session.query(Sample).filter(Sample.external_key.in_(request.keys)).order_by(Sample.file_id).all()
            )

            if len(samples) == 0:
                logger.error("No samples found in the database.")
                yield GetResponse()
                return

            if len(samples) != len(request.keys):
                logger.error("Not all keys were found in the database.")
                not_found_keys = {s for s in request.keys if s not in [sample.external_key for sample in samples]}
                logger.error(f"Keys: {not_found_keys}")

            current_file = samples[0].file
            samples_per_file: list[Tuple[int, str, int]] = []

            # Iterate over all samples and group them by file, the samples are sorted by file_id (see query above)
            for sample in samples:
                if sample.file_id != current_file.file_id:
                    file_wrapper = get_file_wrapper(
                        dataset.file_wrapper_type,
                        current_file.path,
                        dataset.file_wrapper_config,
                        get_filesystem_wrapper(dataset.filesystem_wrapper_type, dataset.base_path),
                    )
                    yield GetResponse(
                        samples=file_wrapper.get_samples_from_indices([index for index, _, _ in samples_per_file]),
                        keys=[external_key for _, external_key, _ in samples_per_file],
                        labels=[label for _, _, label in samples_per_file],
                    )
                    samples_per_file = [(sample.index, sample.external_key, sample.label)]
                    current_file = sample.file
                else:
                    samples_per_file.append((sample.index, sample.external_key, sample.label))
            file_wrapper = get_file_wrapper(
                dataset.file_wrapper_type,
                current_file.path,
                dataset.file_wrapper_config,
                get_filesystem_wrapper(dataset.filesystem_wrapper_type, dataset.base_path),
            )
            yield GetResponse(
                samples=file_wrapper.get_samples_from_indices([index for index, _, _ in samples_per_file]),
                keys=[external_key for _, external_key, _ in samples_per_file],
                labels=[label for _, _, label in samples_per_file],
            )

    # pylint: disable-next=unused-argument,invalid-name
    def GetNewDataSince(
        self, request: GetNewDataSinceRequest, context: grpc.ServicerContext
    ) -> GetNewDataSinceResponse:
        """Get all new data since the given timestamp.

        Returns:
            GetNewDataSinceResponse: A response containing all external keys since the given timestamp.
        """
        with StorageDatabaseConnection(self.modyn_config) as database:
            session = database.session

            dataset: Dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()

            if dataset is None:
                logger.error(f"Dataset with name {request.dataset_id} does not exist.")
                return GetNewDataSinceResponse()

            timestamp = request.timestamp

            values = (
                session.query(Sample.external_key, File.updated_at, Sample.label)
                .join(File)
                .filter(File.dataset_id == dataset.dataset_id)
                .filter(File.updated_at >= timestamp)
                .all()
            )

            if len(values) == 0:
                logger.info(f"No new data since {timestamp}")
                return GetNewDataSinceResponse()

            return GetNewDataSinceResponse(
                keys=[value[0] for value in values],
                timestamps=[value[1] for value in values],
                labels=[value[2] for value in values],
            )

    def GetDataInInterval(
        self, request: GetDataInIntervalRequest, context: grpc.ServicerContext
    ) -> GetDataInIntervalResponse:
        """Get all data in the given interval.

        Returns:
            GetDataInIntervalResponse: A response containing all external keys in the given interval inclusive.
        """
        with StorageDatabaseConnection(self.modyn_config) as database:
            session = database.session

            dataset: Dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()

            if dataset is None:
                logger.error(f"Dataset with name {request.dataset_id} does not exist.")
                return GetDataInIntervalResponse()

            values = (
                session.query(Sample.external_key, File.updated_at, Sample.label)
                .join(File)
                .filter(File.dataset_id == dataset.dataset_id)
                .filter(File.updated_at >= request.start_timestamp)
                .filter(File.updated_at <= request.end_timestamp)
                .all()
            )

            if len(values) == 0:
                logger.info(f"No data between timestamp {request.start_timestamp} and {request.end_timestamp}")
                return GetDataInIntervalResponse()

            return GetDataInIntervalResponse(
                keys=[value[0] for value in values],
                timestamps=[value[1] for value in values],
                labels=[value[2] for value in values],
            )

    # pylint: disable-next=unused-argument,invalid-name
    def CheckAvailability(
        self, request: DatasetAvailableRequest, context: grpc.ServicerContext
    ) -> DatasetAvailableResponse:
        """Check if a dataset is available in the database.

        Returns:
            DatasetAvailableResponse: True if the dataset is available, False otherwise.
        """
        with StorageDatabaseConnection(self.modyn_config) as database:
            session = database.session

            dataset: Dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()

            if dataset is None:
                logger.error(f"Dataset with name {request.dataset_id} does not exist.")
                return DatasetAvailableResponse(available=False)

            return DatasetAvailableResponse(available=True)

    # pylint: disable-next=unused-argument,invalid-name
    def RegisterNewDataset(
        self, request: RegisterNewDatasetRequest, context: grpc.ServicerContext
    ) -> RegisterNewDatasetResponse:
        """Register a new dataset in the database.

        Returns:
            RegisterNewDatasetResponse: True if the dataset was successfully registered, False otherwise.
        """
        with StorageDatabaseConnection(self.modyn_config) as database:
            success = database.add_dataset(
                request.dataset_id,
                request.base_path,
                request.filesystem_wrapper_type,
                request.file_wrapper_type,
                request.description,
                request.version,
                request.file_wrapper_config,
            )
            return RegisterNewDatasetResponse(success=success)

    # pylint: disable-next=unused-argument,invalid-name
    def GetCurrentTimestamp(self, request: None, context: grpc.ServicerContext) -> GetCurrentTimestampResponse:
        """Get the current timestamp.

        Returns:
            GetCurrentTimestampResponse: The current timestamp.
        """
        return GetCurrentTimestampResponse(timestamp=current_time_millis())

    # pylint: disable-next=unused-argument,invalid-name
    def DeleteDataset(self, request: DatasetAvailableRequest, context: grpc.ServicerContext) -> DeleteDatasetResponse:
        """Delete a dataset from the database.

        Returns:
            DeleteDatasetResponse: True if the dataset was successfully deleted, False otherwise.
        """
        with StorageDatabaseConnection(self.modyn_config) as database:
            success = database.delete_dataset(request.dataset_id)
            return DeleteDatasetResponse(success=success)
