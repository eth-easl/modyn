# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: storage.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rstorage.proto\x12\x07storage\"\x1a\n\nGetRequest\x12\x0c\n\x04keys\x18\x01 \x03(\t\"\x1c\n\x0bGetResponse\x12\r\n\x05value\x18\x01 \x03(\x0c\")\n\nPutRequest\x12\x0c\n\x04keys\x18\x01 \x03(\t\x12\r\n\x05value\x18\x02 \x03(\x0c\"\r\n\x0bPutResponse2q\n\x07Storage\x12\x32\n\x03Get\x12\x13.storage.GetRequest\x1a\x14.storage.GetResponse\"\x00\x12\x32\n\x03Put\x12\x13.storage.PutRequest\x1a\x14.storage.PutResponse\"\x00\x62\x06proto3')  # noqa: E501

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'storage_pb2', globals())
if not _descriptor._USE_C_DESCRIPTORS:

    DESCRIPTOR._options = None
    _GETREQUEST._serialized_start = 26
    _GETREQUEST._serialized_end = 52
    _GETRESPONSE._serialized_start = 54
    _GETRESPONSE._serialized_end = 82
    _PUTREQUEST._serialized_start = 84
    _PUTREQUEST._serialized_end = 125
    _PUTRESPONSE._serialized_start = 127
    _PUTRESPONSE._serialized_end = 140
    _STORAGE._serialized_start = 142
    _STORAGE._serialized_end = 255
# @@protoc_insertion_point(module_scope)
