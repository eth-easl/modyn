# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: model_storage.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x13model_storage.proto\x12\x13modyn.model_storage\"S\n\x14RegisterModelRequest\x12\x13\n\x0bpipeline_id\x18\x01 \x01(\x05\x12\x12\n\ntrigger_id\x18\x02 \x01(\x05\x12\x12\n\nmodel_path\x18\x03 \x01(\t\":\n\x15RegisterModelResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x10\n\x08model_id\x18\x02 \x01(\x05\"%\n\x11\x46\x65tchModelRequest\x12\x10\n\x08model_id\x18\x01 \x01(\x05\"7\n\x12\x46\x65tchModelResponse\x12\r\n\x05valid\x18\x01 \x01(\x08\x12\x12\n\nmodel_path\x18\x02 \x01(\t2\xd9\x01\n\x0cModelStorage\x12h\n\rRegisterModel\x12).modyn.model_storage.RegisterModelRequest\x1a*.modyn.model_storage.RegisterModelResponse\"\x00\x12_\n\nFetchModel\x12&.modyn.model_storage.FetchModelRequest\x1a\'.modyn.model_storage.FetchModelResponse\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'model_storage_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _REGISTERMODELREQUEST._serialized_start=44
  _REGISTERMODELREQUEST._serialized_end=127
  _REGISTERMODELRESPONSE._serialized_start=129
  _REGISTERMODELRESPONSE._serialized_end=187
  _FETCHMODELREQUEST._serialized_start=189
  _FETCHMODELREQUEST._serialized_end=226
  _FETCHMODELRESPONSE._serialized_start=228
  _FETCHMODELRESPONSE._serialized_end=283
  _MODELSTORAGE._serialized_start=286
  _MODELSTORAGE._serialized_end=503
# @@protoc_insertion_point(module_scope)
