# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: selector.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0eselector.proto\x12\x08selector\"\x07\n\x05\x45mpty\"\x1b\n\nJsonString\x12\r\n\x05value\x18\x01 \x01(\t\"Z\n\x11\x44\x61taInformRequest\x12\x13\n\x0bpipeline_id\x18\x01 \x01(\x05\x12\x0c\n\x04keys\x18\x02 \x03(\t\x12\x12\n\ntimestamps\x18\x03 \x03(\x03\x12\x0e\n\x06labels\x18\x04 \x03(\x03\"%\n\x0fTriggerResponse\x12\x12\n\ntrigger_id\x18\x01 \x01(\x05\"`\n\x17RegisterPipelineRequest\x12\x13\n\x0bnum_workers\x18\x01 \x01(\x05\x12\x30\n\x12selection_strategy\x18\x02 \x01(\x0b\x32\x14.selector.JsonString\"\'\n\x10PipelineResponse\x12\x13\n\x0bpipeline_id\x18\x01 \x01(\x05\"O\n\x11GetSamplesRequest\x12\x13\n\x0bpipeline_id\x18\x01 \x01(\x05\x12\x12\n\ntrigger_id\x18\x02 \x01(\x05\x12\x11\n\tworker_id\x18\x03 \x01(\x05\"T\n\x0fSamplesResponse\x12\x1f\n\x17training_samples_subset\x18\x01 \x03(\t\x12 \n\x18training_samples_weights\x18\x02 \x03(\x02\"D\n\x19GetNumberOfSamplesRequest\x12\x13\n\x0bpipeline_id\x18\x01 \x01(\x05\x12\x12\n\ntrigger_id\x18\x02 \x01(\x05\"-\n\x16NumberOfSamplesResponse\x12\x13\n\x0bnum_samples\x18\x01 \x01(\x05\x32\xaf\x03\n\x08Selector\x12T\n\x11register_pipeline\x12!.selector.RegisterPipelineRequest\x1a\x1a.selector.PipelineResponse\"\x00\x12W\n\x1bget_sample_keys_and_weights\x12\x1b.selector.GetSamplesRequest\x1a\x19.selector.SamplesResponse\"\x00\x12=\n\x0binform_data\x12\x1b.selector.DataInformRequest\x1a\x0f.selector.Empty\"\x00\x12S\n\x17inform_data_and_trigger\x12\x1b.selector.DataInformRequest\x1a\x19.selector.TriggerResponse\"\x00\x12`\n\x15get_number_of_samples\x12#.selector.GetNumberOfSamplesRequest\x1a .selector.NumberOfSamplesResponse\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'selector_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _EMPTY._serialized_start=28
  _EMPTY._serialized_end=35
  _JSONSTRING._serialized_start=37
  _JSONSTRING._serialized_end=64
  _DATAINFORMREQUEST._serialized_start=66
  _DATAINFORMREQUEST._serialized_end=156
  _TRIGGERRESPONSE._serialized_start=158
  _TRIGGERRESPONSE._serialized_end=195
  _REGISTERPIPELINEREQUEST._serialized_start=197
  _REGISTERPIPELINEREQUEST._serialized_end=293
  _PIPELINERESPONSE._serialized_start=295
  _PIPELINERESPONSE._serialized_end=334
  _GETSAMPLESREQUEST._serialized_start=336
  _GETSAMPLESREQUEST._serialized_end=415
  _SAMPLESRESPONSE._serialized_start=417
  _SAMPLESRESPONSE._serialized_end=501
  _GETNUMBEROFSAMPLESREQUEST._serialized_start=503
  _GETNUMBEROFSAMPLESREQUEST._serialized_end=571
  _NumberOfSamplesResponse._serialized_start=573
  _NumberOfSamplesResponse._serialized_end=618
  _SELECTOR._serialized_start=621
  _SELECTOR._serialized_end=1052
# @@protoc_insertion_point(module_scope)
