# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ptmp.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\nptmp.proto\x12\x04ptmp\"@\n\x1bPostTrainingMetadataRequest\x12\x13\n\x0btraining_id\x18\x01 \x01(\x05\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\t\"\x1e\n\x1cPostTrainingMetadataResponse2\x87\x01\n\x1dPostTrainingMetadataProcessor\x12\x66\n\x1bProcessPostTrainingMetadata\x12!.ptmp.PostTrainingMetadataRequest\x1a\".ptmp.PostTrainingMetadataResponse\"\x00\x62\x06proto3')  # noqa: E501

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ptmp_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _POSTTRAININGMETADATAREQUEST._serialized_start=20
  _POSTTRAININGMETADATAREQUEST._serialized_end=84
  _POSTTRAININGMETADATARESPONSE._serialized_start=86
  _POSTTRAININGMETADATARESPONSE._serialized_end=116
  _POSTTRAININGMETADATAPROCESSOR._serialized_start=119
  _POSTTRAININGMETADATAPROCESSOR._serialized_end=254
# @@protoc_insertion_point(module_scope)
