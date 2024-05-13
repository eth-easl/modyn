import json
import os
import pathlib
import random
import shutil
import time
from typing import Optional

import grpc
import modyn.storage.internal.grpc.generated.storage_pb2 as storage_pb2
import numpy as np
import pandas as pd
import yaml
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.utils import ModelStorageStrategyConfig
from modyn.selector.internal.grpc.generated.selector_pb2 import JsonString as SelectorJsonString
from modyn.selector.internal.grpc.generated.selector_pb2 import StrategyConfig
from modyn.storage.internal.grpc.generated.storage_pb2 import (
    DatasetAvailableRequest,
    GetDatasetSizeRequest,
    GetDatasetSizeResponse,
    RegisterNewDatasetRequest,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.utils import grpc_connection_established
from PIL import Image

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))
MODYN_CONFIG_PATH = pathlib.Path(
    os.getenv("MODYN_CONFIG_PATH", SCRIPT_PATH.parent.parent / "modyn" / "config" / "examples")
)
MODYN_CONFIG_FILE = MODYN_CONFIG_PATH / "modyn_config.yaml"

MODYNCLIENT_CONFIG_PATH = pathlib.Path(
    os.getenv("MODYNCLIENT_CONFIG_PATH", SCRIPT_PATH.parent.parent / "modynclient" / "config" / "examples")
)
MODYN_DATASET_PATH = pathlib.Path(os.getenv("MODYN_DATASET_PATH", pathlib.Path("/app") / "storage" / "datasets"))

CLIENT_CONFIG_FILE = MODYNCLIENT_CONFIG_PATH / "modyn_client_config_container.yaml"
MNIST_CONFIG_FILE = MODYNCLIENT_CONFIG_PATH / "mnist.yaml"
DUMMY_CONFIG_FILE = MODYNCLIENT_CONFIG_PATH / "dummy.yaml"
CLIENT_ENTRYPOINT = SCRIPT_PATH.parent.parent / "modynclient" / "client" / "modyn-client"
NEW_DATASET_TIMEOUT = 600

DEFAULT_SELECTION_STRATEGY = {"name": "NewDataStrategy", "maximum_keys_in_memory": 10}
DEFAULT_MODEL_STORAGE_CONFIG = {"full_model_strategy": {"name": "PyTorchFullModel"}}


def get_minimal_pipeline_config(
    num_workers: int = 1,
    strategy_config: dict = DEFAULT_SELECTION_STRATEGY,
    model_storage_config: dict = DEFAULT_MODEL_STORAGE_CONFIG,
) -> dict:
    return {
        "pipeline": {"name": "Test"},
        "model": {"id": "ResNet18"},
        "model_storage": model_storage_config,
        "training": {
            "gpus": 1,
            "device": "cpu",
            "dataloader_workers": num_workers,
            "use_previous_model": True,
            "initial_model": "random",
            "initial_pass": {"activated": False},
            "learning_rate": 0.1,
            "batch_size": 42,
            "optimizers": [
                {"name": "default1", "algorithm": "SGD", "source": "PyTorch", "param_groups": [{"module": "model"}]},
            ],
            "optimization_criterion": {"name": "CrossEntropyLoss"},
            "checkpointing": {"activated": False},
            "selection_strategy": strategy_config,
        },
        "data": {"dataset_id": "test", "bytes_parser_function": "def bytes_parser_function(x):\n\treturn x"},
        "trigger": {"id": "DataAmountTrigger", "trigger_config": {"data_points_for_trigger": 1}},
    }


def load_config_from_file(config_file: pathlib.Path) -> dict:
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_modyn_config() -> dict:
    return load_config_from_file(MODYN_CONFIG_FILE)


def init_metadata_db(modyn_config: dict) -> None:
    with MetadataDatabaseConnection(modyn_config) as database:
        database.create_tables()


def register_pipeline(pipeline_config: dict, modyn_config: dict) -> int:
    num_workers: int = pipeline_config["training"]["dataloader_workers"]
    if num_workers < 0:
        raise ValueError(f"Tried to register training with {num_workers} workers.")

    if "config" in pipeline_config["model"]:
        model_config = json.dumps(pipeline_config["model"]["config"])
    else:
        model_config = "{}"

    model_storage_config = pipeline_config["model_storage"]
    full_model_strategy = ModelStorageStrategyConfig.from_config(
        get_model_strategy(model_storage_config["full_model_strategy"])
    )
    incremental_model_strategy_config: Optional[StrategyConfig] = None
    full_model_interval: Optional[int] = None
    if "incremental_model_strategy" in model_storage_config:
        incremental_strategy = model_storage_config["incremental_model_strategy"]
        incremental_model_strategy_config = get_model_strategy(incremental_strategy)
        full_model_interval = (
            incremental_strategy["full_model_interval"] if "full_model_interval" in incremental_strategy else None
        )

    incremental_model_strategy: Optional[ModelStorageStrategyConfig] = None
    if incremental_model_strategy_config is not None:
        incremental_model_strategy = ModelStorageStrategyConfig.from_config(incremental_model_strategy_config)

    with MetadataDatabaseConnection(modyn_config) as database:
        pipeline_id = database.register_pipeline(
            num_workers=num_workers,
            model_class_name=pipeline_config["model"]["id"],
            model_config=model_config,
            amp=pipeline_config["training"]["amp"] if "amp" in pipeline_config["training"] else False,
            selection_strategy=json.dumps(pipeline_config["training"]["selection_strategy"]),
            full_model_strategy=full_model_strategy,
            incremental_model_strategy=incremental_model_strategy,
            full_model_interval=full_model_interval,
        )

    return pipeline_id


def get_model_strategy(strategy_config: dict) -> StrategyConfig:
    return StrategyConfig(
        name=strategy_config["name"],
        zip=strategy_config["zip"] if "zip" in strategy_config else None,
        zip_algorithm=strategy_config["zip_algorithm"] if "zip_algorithm" in strategy_config else None,
        config=SelectorJsonString(value=json.dumps(strategy_config)),
    )


def get_server_address(server_name: str) -> str:
    config = get_modyn_config()
    if server_name not in config:
        raise ValueError(f"{server_name} is not a server defined in modyn config!")
    return f"{config[server_name]['hostname']}:{config[server_name]['port']}"


def connect_to_server(server_name: str) -> grpc.Channel:
    server_address = get_server_address(server_name)
    server_channel = grpc.insecure_channel(server_address)

    if not grpc_connection_established(server_channel) or server_channel is None:
        raise ConnectionError(f"Could not establish gRPC connection to {server_name} at {server_channel}.")

    return server_channel


class DatasetHelper:
    def __init__(
        self,
        dataset_id: str,
        dataset_size: int = 10,
        dataset_dir: pathlib.Path = MODYN_DATASET_PATH,
        desc: str = "new dataset",
        file_wrapper_config: dict = {"file_extension": ".png", "label_file_extension": ".txt"},
        file_wrapper_type: str = "SingleSampleFileWrapper",
        filesystem_wrapper_type: str = "LocalFilesystemWrapper",
        file_watcher_interval: int = 5,
        version: str = "0.1.0",
    ) -> None:
        self.dataset_id = dataset_id
        self.dataset_size = dataset_size
        self.dataset_path = dataset_dir / dataset_id
        self.desc = desc
        self.file_wrapper_config = file_wrapper_config
        self.file_wrapper_type = file_wrapper_type
        self.filesystem_wrapper_type = filesystem_wrapper_type
        self.file_watcher_interval = file_watcher_interval
        self.version = version

        self.storage_channel = connect_to_server("storage")
        self.storage = StorageStub(self.storage_channel)

    def check_get_current_timestamp(self) -> None:
        empty = storage_pb2.google_dot_protobuf_dot_empty__pb2.Empty()
        response = self.storage.GetCurrentTimestamp(empty)
        assert response.timestamp > 0, "Timestamp is not valid."

    def create_dataset_dir(self) -> None:
        pathlib.Path(self.dataset_path).mkdir(parents=True, exist_ok=True)

    def cleanup_dataset_dir(self) -> None:
        shutil.rmtree(self.dataset_path)

    def cleanup_storage_database(self) -> None:
        request = DatasetAvailableRequest(dataset_id=self.dataset_id)
        response = self.storage.DeleteDataset(request)
        assert response.success, "Could not cleanup storage database."

    def check_dataset_availability(self) -> None:
        request = DatasetAvailableRequest(dataset_id=self.dataset_id)
        response = self.storage.CheckAvailability(request)

        assert response.available, "Dataset is not available."

    def wait_for_dataset(self, expected_size: int) -> None:
        time.sleep(NEW_DATASET_TIMEOUT)
        request = GetDatasetSizeRequest(dataset_id=self.dataset_id)
        response: GetDatasetSizeResponse = self.storage.GetDatasetSize(request)

        assert response.success, f"Dataset is not available after {NEW_DATASET_TIMEOUT} sec."
        assert response.num_keys >= expected_size

    def register_new_dataset(self) -> None:
        request = RegisterNewDatasetRequest(
            base_path=str(self.dataset_path),
            dataset_id=self.dataset_id,
            description=self.desc,
            file_wrapper_config=json.dumps(self.file_wrapper_config),
            file_wrapper_type=self.file_wrapper_type,
            filesystem_wrapper_type=self.filesystem_wrapper_type,
            file_watcher_interval=self.file_watcher_interval,
            version=self.version,
        )

        response = self.storage.RegisterNewDataset(request)

        assert response.success, "Could not register new dataset."

    def setup_dataset(self) -> None:
        self.check_get_current_timestamp()  # Check if the storage service is available.
        self.create_dataset_dir()
        self.create_dataset()
        self.register_new_dataset()
        self.check_dataset_availability()  # Check if the dataset is available.
        self.wait_for_dataset(self.dataset_size)

    def create_dataset(self) -> None:
        pass


class ImageDatasetHelper(DatasetHelper):
    def __init__(
        self,
        dataset_id: str = "image_dataset",
        dataset_size: int = 10,
        dataset_dir: pathlib.Path = MODYN_DATASET_PATH,
        desc: str = "Test dataset for integration tests.",
        first_added_images: list = [],
    ) -> None:
        super().__init__(
            dataset_id,
            dataset_size,
            dataset_dir,
            desc,
            {"file_extension": ".png", "label_file_extension": ".txt"},
        )
        self.first_added_images = first_added_images

    def create_random_image(self) -> Image:
        image = Image.new("RGB", (100, 100))
        random_x = random.randint(0, 99)
        random_y = random.randint(0, 99)

        random_r = random.randint(0, 254)
        random_g = random.randint(0, 254)
        random_b = random.randint(0, 254)

        image.putpixel((random_x, random_y), (random_r, random_g, random_b))

        return image

    def add_image_to_dataset(self, image: Image, name: str) -> None:
        image.save(self.dataset_path / name)

    def add_images_to_dataset(self, start_number: int, end_number: int, images_added: list[bytes]) -> None:
        for i in range(start_number, end_number):
            image = self.create_random_image()
            self.add_image_to_dataset(image, f"image_{i}.png")
            images_added.append(image.tobytes())
            with open(self.dataset_path / f"image_{i}.txt", "w") as label_file:
                label_file.write(f"{i}")

    def create_dataset(self) -> None:
        self.add_images_to_dataset(0, self.dataset_size, self.first_added_images)  # Add images to the dataset.


class TinyDatasetHelper(DatasetHelper):
    def __init__(
        self,
        dataset_id: str = "tiny_dataset",
        dataset_size: int = 10,
        dataset_dir: pathlib.Path = MODYN_DATASET_PATH,
        desc: str = "Tiny dataset for integration tests.",
    ) -> None:
        super().__init__(
            dataset_id, dataset_size, dataset_dir, desc, {"file_extension": ".csv", "label_file_extension": ".txt"}
        )

    def create_dataset(self) -> None:
        rng = np.random.default_rng()
        for n in range(self.dataset_size):
            a = rng.random(size=(1, 2), dtype=np.float32)
            df = pd.DataFrame(a)
            df.to_csv(self.dataset_path / f"tensor_{n}.csv", index=False)
            with open(self.dataset_path / f"tensor_{n}.txt", "w") as label_file:
                label_file.write(f"{n % 2}")
