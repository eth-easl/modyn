# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: metadata.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

# pylint: skip-file


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0emetadata.proto\x12\x08metadata\"A\n\x0fRegisterRequest\x12\x19\n\x11training_set_size\x18\x01 \x01(\x05\x12\x13\n\x0bnum_workers\x18\x02 \x01(\x05\"\'\n\x10RegisterResponse\x12\x13\n\x0btraining_id\x18\x01 \x01(\x05\")\n\x12GetTrainingRequest\x12\x13\n\x0btraining_id\x18\x01 \x01(\x05\"B\n\x10TrainingResponse\x12\x19\n\x11training_set_size\x18\x01 \x01(\x05\x12\x13\n\x0bnum_workers\x18\x02 \x01(\x05\"5\n\x10GetByKeysRequest\x12\x13\n\x0btraining_id\x18\x01 \x01(\x05\x12\x0c\n\x04keys\x18\x02 \x03(\t\"\"\n\x11GetByQueryRequest\x12\r\n\x05query\x18\x01 \x01(\t\"V\n\x0bGetResponse\x12\x0c\n\x04keys\x18\x01 \x03(\t\x12\x0e\n\x06scores\x18\x02 \x03(\x02\x12\x0c\n\x04\x64\x61ta\x18\x03 \x03(\t\x12\x0c\n\x04seen\x18\x04 \x03(\x08\x12\r\n\x05label\x18\x05 \x03(\x05\"\x1f\n\x0fGetKeysResponse\x12\x0c\n\x04keys\x18\x01 \x03(\t\"j\n\nSetRequest\x12\x13\n\x0btraining_id\x18\x01 \x01(\x05\x12\x0c\n\x04keys\x18\x02 \x03(\t\x12\x0e\n\x06scores\x18\x03 \x03(\x02\x12\x0c\n\x04seen\x18\x04 \x03(\x08\x12\r\n\x05label\x18\x05 \x03(\x05\x12\x0c\n\x04\x64\x61ta\x18\x06 \x03(\t\"\r\n\x0bSetResponse\"$\n\rDeleteRequest\x12\x13\n\x0btraining_id\x18\x01 \x01(\x05\"\x10\n\x0e\x44\x65leteResponse2\xf5\x03\n\x08Metadata\x12@\n\tGetByKeys\x12\x1a.metadata.GetByKeysRequest\x1a\x15.metadata.GetResponse\"\x00\x12\x42\n\nGetByQuery\x12\x1b.metadata.GetByQueryRequest\x1a\x15.metadata.GetResponse\"\x00\x12J\n\x0eGetKeysByQuery\x12\x1b.metadata.GetByQueryRequest\x1a\x19.metadata.GetKeysResponse\"\x00\x12\x34\n\x03Set\x12\x14.metadata.SetRequest\x1a\x15.metadata.SetResponse\"\x00\x12\x45\n\x0e\x44\x65leteTraining\x12\x17.metadata.DeleteRequest\x1a\x18.metadata.DeleteResponse\"\x00\x12K\n\x10RegisterTraining\x12\x19.metadata.RegisterRequest\x1a\x1a.metadata.RegisterResponse\"\x00\x12M\n\x0fGetTrainingInfo\x12\x1c.metadata.GetTrainingRequest\x1a\x1a.metadata.TrainingResponse\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'metadata_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _REGISTERREQUEST._serialized_start=28
  _REGISTERREQUEST._serialized_end=93
  _REGISTERRESPONSE._serialized_start=95
  _REGISTERRESPONSE._serialized_end=134
  _GETTRAININGREQUEST._serialized_start=136
  _GETTRAININGREQUEST._serialized_end=177
  _TRAININGRESPONSE._serialized_start=179
  _TRAININGRESPONSE._serialized_end=245
  _GETBYKEYSREQUEST._serialized_start=247
  _GETBYKEYSREQUEST._serialized_end=300
  _GETBYQUERYREQUEST._serialized_start=302
  _GETBYQUERYREQUEST._serialized_end=336
  _GETRESPONSE._serialized_start=338
  _GETRESPONSE._serialized_end=424
  _GETKEYSRESPONSE._serialized_start=426
  _GETKEYSRESPONSE._serialized_end=457
  _SETREQUEST._serialized_start=459
  _SETREQUEST._serialized_end=565
  _SETRESPONSE._serialized_start=567
  _SETRESPONSE._serialized_end=580
  _DELETEREQUEST._serialized_start=582
  _DELETEREQUEST._serialized_end=618
  _DELETERESPONSE._serialized_start=620
  _DELETERESPONSE._serialized_end=636
  _METADATA._serialized_start=639
  _METADATA._serialized_end=1140
# @@protoc_insertion_point(module_scope)
