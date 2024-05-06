# How to generate the python files from the proto files

This assumes python 3.6+ is installed.

## Install the grpcio-tools package

`pip install grpcio-tools==1.63.0`

## Generate the python files

First move to the directory where you want to generate the python files.

Then run the following command:
`python -m grpc_tools.protoc -I../../../../protos --python_out=. --grpc_python_out=. --mypy_out=. ../../../../protos/[component_name].proto`

This will generate the following files:
- [component_name]_pb2.py
- [component_name]_pb2_grpc.py
- [component_name]_pb2.pyi

Please be aware that the relative path to the proto file is important.

Due to weirdness of the gRPC compiler, the import in the `pb2_grpc.py` file might be missing the module prefix, which needs to be fixed manually. For example, 
```
import storage_pb2 as storage__pb2
````
needs to be changes to 
```
import modyn.storage.storage_pb2 as storage__pb2
```


For more information about the protoc compiler, please refer to the [official documentation](https://grpc.io/docs/protoc-installation/) and the [Basics tutorial](https://grpc.io/docs/languages/python/basics/).