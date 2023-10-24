# Storage

This is the storage submodule.

Storage is the abstraction layer for the data storage.
It is responsible for retrieving samples from the actual storage systems and providing them to the GPU nodes for training upon request.
The storage component is started using `modyn-storage config.yaml`.
The binary should be in PATH after building the `modyn` module.
The configuration file describes the system setup.

---

## How the storage abstraction works

The storage abstraction works with the concept of datasets
Each dataset is identified by a unique name and describes a set of files that are stored in a storage system (for more information see the subsection on [How the storage database works](#how-the-storage-database-works))
Each file may contain one or more samples
A dataset is defined by a filesystem wrapper and a file wrapper
The filesystem wrapper describes how to access the underlying filesystem, while the file wrapper describes how to access the samples within the file
The storage abstraction is designed to be flexible and allow for different storage systems and file formats.

### Filesystem wrappers

The following filesystem wrappers are currently implemented:

- `LocalFilesystemWrapper`: Accesses the local filesystem

Future filesystem wrappers may include:

- `s3`: Accesses the Amazon S3 storage system
- `gcs`: Accesses the Google Cloud Storage system

See the `modyn/storage/include/internal/filesystem_wrapper` directory for more information.

**How to add a new filesystem wrapper:**

To add a new filesystem wrapper, you need to implement the `FilesystemWrapper` abstract class
The class is defined in `modyn/storage/include/internal/filesystem_wrapper/filesystem_wrapper.hpp`.

### File wrappers

The following file wrappers are currently implemented:

- `SingleSampleFileWrapper`: Each file contains a single sample
- `BinaryFileWrapper`: Each file contains columns and row in a binary format
- `CsvFileWrapper`: Each file contains columns and rows in a csv format

Future file wrappers may include:

- `tfrecord`: Each file contains multiple samples in the [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format
- `hdf5`: Each file contains multiple samples in the [HDF5](https://www.hdfgroup.org/solutions/hdf5/) format
- `parquet`: Each file contains multiple samples in the [Parquet](https://parquet.apache.org/) format

See the `modyn/storage/include/internal/file_wrapper` directory for more information.

**How to add a new file wrapper:**

To add a new file wrapper, you need to implement the `FileWrapper` class.
The class is defined in `modyn/storage/include/internal/file_wrapper/file_wrapper.hpp`.

---

## How to add a dataset

There are two ways to add a dataset to the storage abstraction:

- Define the dataset in the configuration file and start the storage component using `modyn-storage path/to/config.yaml`.
  If the dataset is not yet in the database, it will be added automatically.
  If the dataset is already in the database, the database entry will be updated.
- Register the dataset using the grpc interface.
  The grpc interface is defined in `modyn/protos/storage.proto`.
  The call is `RegisterNewDataset`.

---

## How to add a file to a dataset (NewFileWatcher)

A file is added to the storage abstraction automatically when the file is created in the underlying storage system.
The storage abstraction will periodically check the underlying storage system for new files.
If a new file is found, it will be added to the database.
The component that is responsible for checking the underlying storage systems is called the `FileWatchdog`.
The `FileWatchdog` is started automatically when the storage component is started.
The `FileWatchdog` is defined in `modyn/storage/include/internal/file_watcher/file_watchdog.hpp`.
The `FileWatchdog` periodically checks for each dataset if there are new files in the underlying storage system with an instance of a `FileWatcher` as defined in `modyn/storage/include/internal/file_watcher/file_watcher.hpp`.
If a new file is found, it and the samples in the file are added to the database.
Files and samples are expected to be added by a separate component or an altogether different system.
The `Storage` component is only responsible for checking for new files and adding them to the database as well as providing the samples to the GPU nodes.
It is thus a read-only component.

---

## How the storage database works

The storage abstraction uses a database to store information about the datasets.
The database contains the following tables:

- `datasets`: Contains information about the datasets
  - `dataset_id`: The id of the dataset (primary key)
  - `name`: The name of the dataset
  - `description`: A description of the dataset
  - `filesystem_wrapper_type`: The name of the filesystem wrapper
  - `file_wrapper_type`: The name of the file wrapper
  - `base_path`: The base path of the dataset
- `files`: Contains information about the files in the datasets
  - `file_id`: The id of the file (primary key)
  - `dataset_id`: The id of the dataset (foreign key to `datasets.dataset_id`)
  - `path`: The path of the file
  - `created_at`: The timestamp when the file was created
  - `updated_at`: The timestamp when the file was updated
  - `number_of_samples`: The number of samples in the file
- `samples`: Contains information about the samples in the files
  - `sample_id`: The id of the sample (primary key)
  - `file_id`: The id of the file (foreign key to `files.file_id`)
  - `index`: The index of the sample in the file
