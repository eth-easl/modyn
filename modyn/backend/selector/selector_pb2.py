# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: selector.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0eselector.proto\x12\x08selector\"I\n\x17RegisterTrainingRequest\x12\x19\n\x11training_set_size\x18\x01 \x01(\x05\x12\x13\n\x0bnum_workers\x18\x02 \x01(\x05\"\'\n\x10TrainingResponse\x12\x13\n\x0btraining_id\x18\x01 \x01(\x05\"X\n\x11GetSamplesRequest\x12\x13\n\x0btraining_id\x18\x01 \x01(\x05\x12\x1b\n\x13training_set_number\x18\x02 \x01(\x05\x12\x11\n\tworker_id\x18\x03 \x01(\x05\"2\n\x0fSamplesResponse\x12\x1f\n\x17training_samples_subset\x18\x01 \x03(\t2\xad\x01\n\x08Selector\x12T\n\x11register_training\x12!.selector.RegisterTrainingRequest\x1a\x1a.selector.TrainingResponse\"\x00\x12K\n\x0fget_sample_keys\x12\x1b.selector.GetSamplesRequest\x1a\x19.selector.SamplesResponse\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'selector_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _REGISTERTRAININGREQUEST._serialized_start=28
  _REGISTERTRAININGREQUEST._serialized_end=101
  _TRAININGRESPONSE._serialized_start=103
  _TRAININGRESPONSE._serialized_end=142
  _GETSAMPLESREQUEST._serialized_start=144
  _GETSAMPLESREQUEST._serialized_end=232
  _SAMPLESRESPONSE._serialized_start=234
  _SAMPLESRESPONSE._serialized_end=284
  _SELECTOR._serialized_start=287
  _SELECTOR._serialized_end=460
# @@protoc_insertion_point(module_scope)
