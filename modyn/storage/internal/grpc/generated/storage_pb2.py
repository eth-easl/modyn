# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: storage.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rstorage.proto\x12\rmodyn.storage\x1a\x1bgoogle/protobuf/empty.proto\".\n\nGetRequest\x12\x12\n\ndataset_id\x18\x01 \x01(\t\x12\x0c\n\x04keys\x18\x02 \x03(\t\"*\n\x0bGetResponse\x12\r\n\x05\x63hunk\x18\x01 \x01(\x0c\x12\x0c\n\x04keys\x18\x02 \x03(\t\"?\n\x16GetNewDataSinceRequest\x12\x12\n\ndataset_id\x18\x01 \x01(\t\x12\x11\n\ttimestamp\x18\x02 \x01(\x03\"\'\n\x17GetNewDataSinceResponse\x12\x0c\n\x04keys\x18\x01 \x03(\t\"^\n\x18GetDataInIntervalRequest\x12\x12\n\ndataset_id\x18\x01 \x01(\t\x12\x17\n\x0fstart_timestamp\x18\x02 \x01(\x03\x12\x15\n\rend_timestamp\x18\x03 \x01(\x03\")\n\x19GetDataInIntervalResponse\x12\x0c\n\x04keys\x18\x01 \x03(\t\"-\n\x17\x44\x61tasetAvailableRequest\x12\x12\n\ndataset_id\x18\x01 \x01(\t\"-\n\x18\x44\x61tasetAvailableResponse\x12\x11\n\tavailable\x18\x01 \x01(\x08\"\xc1\x01\n\x19RegisterNewDatasetRequest\x12\x12\n\ndataset_id\x18\x01 \x01(\t\x12\x1f\n\x17\x66ilesystem_wrapper_type\x18\x02 \x01(\t\x12\x19\n\x11\x66ile_wrapper_type\x18\x03 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x04 \x01(\t\x12\x11\n\tbase_path\x18\x05 \x01(\t\x12\x0f\n\x07version\x18\x06 \x01(\t\x12\x1b\n\x13\x66ile_wrapper_config\x18\x07 \x01(\t\"-\n\x1aRegisterNewDatasetResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\"0\n\x1bGetCurrentTimestampResponse\x12\x11\n\ttimestamp\x18\x01 \x01(\x03\x32\xcf\x04\n\x07Storage\x12@\n\x03Get\x12\x19.modyn.storage.GetRequest\x1a\x1a.modyn.storage.GetResponse\"\x00\x30\x01\x12\x64\n\x0fGetNewDataSince\x12%.modyn.storage.GetNewDataSinceRequest\x1a&.modyn.storage.GetNewDataSinceResponse\"\x00\x30\x01\x12j\n\x11GetDataInInterval\x12\'.modyn.storage.GetDataInIntervalRequest\x1a(.modyn.storage.GetDataInIntervalResponse\"\x00\x30\x01\x12\x66\n\x11\x43heckAvailability\x12&.modyn.storage.DatasetAvailableRequest\x1a\'.modyn.storage.DatasetAvailableResponse\"\x00\x12k\n\x12RegisterNewDataset\x12(.modyn.storage.RegisterNewDatasetRequest\x1a).modyn.storage.RegisterNewDatasetResponse\"\x00\x12[\n\x13GetCurrentTimestamp\x12\x16.google.protobuf.Empty\x1a*.modyn.storage.GetCurrentTimestampResponse\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'storage_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _GETREQUEST._serialized_start=61
  _GETREQUEST._serialized_end=107
  _GETRESPONSE._serialized_start=109
  _GETRESPONSE._serialized_end=151
  _GETNEWDATASINCEREQUEST._serialized_start=153
  _GETNEWDATASINCEREQUEST._serialized_end=216
  _GETNEWDATASINCERESPONSE._serialized_start=218
  _GETNEWDATASINCERESPONSE._serialized_end=257
  _GETDATAININTERVALREQUEST._serialized_start=259
  _GETDATAININTERVALREQUEST._serialized_end=353
  _GETDATAININTERVALRESPONSE._serialized_start=355
  _GETDATAININTERVALRESPONSE._serialized_end=396
  _DATASETAVAILABLEREQUEST._serialized_start=398
  _DATASETAVAILABLEREQUEST._serialized_end=443
  _DATASETAVAILABLERESPONSE._serialized_start=445
  _DATASETAVAILABLERESPONSE._serialized_end=490
  _REGISTERNEWDATASETREQUEST._serialized_start=493
  _REGISTERNEWDATASETREQUEST._serialized_end=686
  _REGISTERNEWDATASETRESPONSE._serialized_start=688
  _REGISTERNEWDATASETRESPONSE._serialized_end=733
  _GETCURRENTTIMESTAMPRESPONSE._serialized_start=735
  _GETCURRENTTIMESTAMPRESPONSE._serialized_end=783
  _STORAGE._serialized_start=786
  _STORAGE._serialized_end=1377
# @@protoc_insertion_point(module_scope)
