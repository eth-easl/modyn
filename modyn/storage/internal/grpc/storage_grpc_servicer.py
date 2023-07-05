"""Storage GRPC servicer."""

import logging
from typing import Iterable, Tuple

import grpc
from modyn.storage.internal.database.models import Dataset, File, Sample
from modyn.storage.internal.database.storage_database_connection import StorageDatabaseConnection
from modyn.storage.internal.database.storage_database_utils import get_file_wrapper, get_filesystem_wrapper

# pylint: disable-next=no-name-in-module
from modyn.storage.internal.grpc.generated.storage_pb2 import (
    DatasetAvailableRequest,
    DatasetAvailableResponse,
    DeleteDataRequest,
    DeleteDataResponse,
    DeleteDatasetResponse,
    GetCurrentTimestampResponse,
    GetDataInIntervalRequest,
    GetDataInIntervalResponse,
    GetDataPerWorkerRequest,
    GetDataPerWorkerResponse,
    GetNewDataSinceRequest,
    GetNewDataSinceResponse,
    GetRequest,
    GetResponse,
    RegisterNewDatasetRequest,
    RegisterNewDatasetResponse,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageServicer
from modyn.utils import current_time_millis, get_partition_for_worker
from sqlalchemy import asc, select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class StorageGRPCServicer(StorageServicer):
    """GRPC servicer for the storage module."""

    def __init__(self, config: dict):
        """Initialize the storage GRPC servicer.

        Args:
            config (dict): Configuration of the storage module.
        """
        self.modyn_config = config
        self._sample_batch_size = self.modyn_config["storage"]["sample_batch_size"]
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
                session.query(Sample).filter(Sample.sample_id.in_(request.keys)).order_by(Sample.file_id).all()
            )

            if len(samples) == 0:
                logger.error("No samples found in the database.")
                yield GetResponse()
                return

            if len(samples) != len(request.keys):
                logger.error("Not all keys were found in the database.")
                not_found_keys = {s for s in request.keys if s not in [sample.sample_id for sample in samples]}
                logger.error(f"Keys: {not_found_keys}")

            current_file_id = samples[0].file_id
            current_file = (
                session.query(File)
                .filter(File.file_id == current_file_id and File.dataset_id == dataset.dataset_id)
                .first()
            )
            samples_per_file: list[Tuple[int, int, int]] = []

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
                        keys=[sample_id for _, sample_id, _ in samples_per_file],
                        labels=[label for _, _, label in samples_per_file],
                    )
                    samples_per_file = [(sample.index, sample.sample_id, sample.label)]
                    current_file_id = sample.file_id
                    current_file = (
                        session.query(File)
                        .filter(File.file_id == current_file_id and File.dataset_id == dataset.dataset_id)
                        .first()
                    )
                else:
                    samples_per_file.append((sample.index, sample.sample_id, sample.label))
            file_wrapper = get_file_wrapper(
                dataset.file_wrapper_type,
                current_file.path,
                dataset.file_wrapper_config,
                get_filesystem_wrapper(dataset.filesystem_wrapper_type, dataset.base_path),
            )
            yield GetResponse(
                samples=file_wrapper.get_samples_from_indices([index for index, _, _ in samples_per_file]),
                keys=[sample_id for _, sample_id, _ in samples_per_file],
                labels=[label for _, _, label in samples_per_file],
            )

    # pylint: disable-next=unused-argument,invalid-name
    def GetNewDataSince(
        self, request: GetNewDataSinceRequest, context: grpc.ServicerContext
    ) -> Iterable[GetNewDataSinceResponse]:
        """Get all new data since the given timestamp.

        Returns:
            GetNewDataSinceResponse: A response containing all external keys since the given timestamp.
        """
        with StorageDatabaseConnection(self.modyn_config) as database:
            session = database.session

            dataset: Dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()

            if dataset is None:
                logger.error(f"Dataset with name {request.dataset_id} does not exist.")
                yield GetNewDataSinceResponse()
                return

            timestamp = request.timestamp

            stmt = (
                select(Sample.sample_id, File.updated_at, Sample.label)
                .join(File, Sample.file_id == File.file_id and Sample.dataset_id == File.dataset_id)
                # Enables batching of results in chunks.
                # See https://docs.sqlalchemy.org/en/20/orm/queryguide/api.html#orm-queryguide-yield-per
                .execution_options(yield_per=self._sample_batch_size)
                .filter(File.dataset_id == dataset.dataset_id)
                .filter(File.updated_at >= timestamp)
                .order_by(asc(File.updated_at), asc(Sample.sample_id))
            )

            for batch in database.session.execute(stmt).partitions():
                if len(batch) > 0:
                    yield GetNewDataSinceResponse(
                        keys=[value[0] for value in batch],
                        timestamps=[value[1] for value in batch],
                        labels=[value[2] for value in batch],
                    )

    def GetDataInInterval(
        self, request: GetDataInIntervalRequest, context: grpc.ServicerContext
    ) -> Iterable[GetDataInIntervalResponse]:
        """Get all data in the given interval.

        Returns:
            GetDataInIntervalResponse: A response containing all external keys in the given interval inclusive.
        """
        with StorageDatabaseConnection(self.modyn_config) as database:
            session = database.session

            dataset: Dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()

            if dataset is None:
                logger.error(f"Dataset with name {request.dataset_id} does not exist.")
                yield GetDataInIntervalResponse()
                return

            stmt = (
                select(Sample.sample_id, File.updated_at, Sample.label)
                .join(File, Sample.file_id == File.file_id and Sample.dataset_id == File.dataset_id)
                # Enables batching of results in chunks.
                # See https://docs.sqlalchemy.org/en/20/orm/queryguide/api.html#orm-queryguide-yield-per
                .execution_options(yield_per=self._sample_batch_size)
                .filter(File.dataset_id == dataset.dataset_id)
                .filter(File.updated_at >= request.start_timestamp)
                .filter(File.updated_at <= request.end_timestamp)
                .order_by(asc(File.updated_at), asc(Sample.sample_id))
            )

            for batch in database.session.execute(stmt).partitions():
                if len(batch) > 0:
                    yield GetDataInIntervalResponse(
                        keys=[value[0] for value in batch],
                        timestamps=[value[1] for value in batch],
                        labels=[value[2] for value in batch],
                    )

    # pylint: disable-next=unused-argument,invalid-name
    def GetDataPerWorker(
        self, request: GetDataPerWorkerRequest, context: grpc.ServicerContext
    ) -> Iterable[GetDataPerWorkerResponse]:
        """Get keys from the given dataset for a worker.

        Returns:
            GetDataPerWorkerResponse: A response containing external keys from the dataset for the worker.
        """
        with StorageDatabaseConnection(self.modyn_config) as database:
            session = database.session

            dataset: Dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()

            if dataset is None:
                logger.error(f"Dataset with name {request.dataset_id} does not exist.")
                yield GetNewDataSinceResponse()
                return

            total_keys = session.query(Sample.sample_id).filter(Sample.dataset_id == dataset.dataset_id).count()
            start_index, limit = get_partition_for_worker(request.worker_id, request.total_workers, total_keys)

            stmt = (
                select(Sample.sample_id)
                # Enables batching of results in chunks.
                # See https://docs.sqlalchemy.org/en/20/orm/queryguide/api.html#orm-queryguide-yield-per
                .execution_options(yield_per=self._sample_batch_size)
                .filter(Sample.dataset_id == dataset.dataset_id)
                .order_by(Sample.sample_id)
                .offset(start_index)
                .limit(limit)
            )

            for batch in database.session.execute(stmt).partitions():
                if len(batch) > 0:
                    yield GetDataPerWorkerResponse(keys=[value[0] for value in batch])

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
                request.ignore_last_timestamp,
                request.file_watcher_interval,
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

    def DeleteData(self, request: DeleteDataRequest, context: grpc.ServicerContext) -> DeleteDataResponse:
        """Delete data from the database.

        Returns:
            DeleteDataResponse: True if the data was successfully deleted, False otherwise.
        """
        with StorageDatabaseConnection(self.modyn_config) as database:
            session = database.session
            dataset: Dataset = session.query(Dataset).filter(Dataset.name == request.dataset_id).first()
            if dataset is None:
                logger.error(f"Dataset with name {request.dataset_id} does not exist.")
                return DeleteDataResponse(success=False)

            file_ids: list[Sample] = (
                session.query(Sample.file_id)
                .filter(Sample.sample_id.in_(request.keys))
                .order_by(Sample.file_id)
                .group_by(Sample.file_id)
                .all()
            )

            for file_id in file_ids:
                file_id = file_id[0]
                file: File = session.query(File).filter(File.file_id == file_id).first()
                if file is None:
                    logger.error(f"Could not find file for dataset {request.dataset_id}")
                    return DeleteDataResponse(success=False)
                file_wrapper = get_file_wrapper(
                    dataset.file_wrapper_type,
                    file.path,
                    dataset.file_wrapper_config,
                    get_filesystem_wrapper(dataset.filesystem_wrapper_type, dataset.base_path),
                )
                samples_to_delete = (
                    session.query(Sample)
                    .filter(Sample.file_id == file.file_id)
                    .filter(Sample.sample_id.in_(request.keys))
                    .all()
                )
                file_wrapper.delete_samples(samples_to_delete)
                file.number_of_samples -= len(samples_to_delete)
                session.commit()

            session.query(Sample).filter(Sample.sample_id.in_(request.keys)).delete()
            session.commit()

            self.remove_empty_files(session, dataset)

            return DeleteDataResponse(success=True)

    def remove_empty_files(self, session: Session, dataset: Dataset) -> None:
        """Delete files that have no samples left."""
        files_to_delete = (
            session.query(File).filter(File.dataset_id == dataset.dataset_id).filter(File.number_of_samples == 0).all()
        )
        for file in files_to_delete:
            file_system_wrapper = get_filesystem_wrapper(dataset.filesystem_wrapper_type, dataset.base_path)
            try:
                file_system_wrapper.delete(file.path)
            except FileNotFoundError:
                logger.debug(
                    f"File {file.path} not found. Might have been deleted \
                        already in the previous step of this method."
                )
            session.query(File).filter(File.file_id == file.file_id).delete()
        session.commit()
