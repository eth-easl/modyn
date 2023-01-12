# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: trainer_server.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14trainer_server.proto\x12\x07trainer\"\x8d\x03\n\x1aRegisterTrainServerRequest\x12\x13\n\x0btraining_id\x18\x01 \x01(\x05\x12\x10\n\x08model_id\x18\x02 \x01(\t\x12\x17\n\x0ftorch_optimizer\x18\x03 \x01(\t\x12\x12\n\nbatch_size\x18\x04 \x01(\x05\x12\x17\n\x0ftorch_criterion\x18\x05 \x01(\t\x12\x31\n\x14\x63riterion_parameters\x18\x06 \x01(\x0b\x32\x13.trainer.JsonString\x12\x31\n\x14optimizer_parameters\x18\x07 \x01(\x0b\x32\x13.trainer.JsonString\x12\x30\n\x13model_configuration\x18\x08 \x01(\x0b\x32\x13.trainer.JsonString\x12 \n\tdata_info\x18\t \x01(\x0b\x32\r.trainer.Data\x12\x30\n\x0f\x63heckpoint_info\x18\n \x01(\x0b\x32\x17.trainer.CheckpointInfo\x12\x16\n\x0etransform_list\x18\x0b \x03(\t\"\x1b\n\nJsonString\x12\r\n\x05value\x18\x01 \x01(\t\".\n\x1bRegisterTrainServerResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\"3\n\x04\x44\x61ta\x12\x12\n\ndataset_id\x18\x01 \x01(\t\x12\x17\n\x0fnum_dataloaders\x18\x02 \x01(\x05\"\x19\n\x17TrainerAvailableRequest\"-\n\x18TrainerAvailableResponse\x12\x11\n\tavailable\x18\x01 \x01(\x08\"F\n\x0e\x43heckpointInfo\x12\x1b\n\x13\x63heckpoint_interval\x18\x01 \x01(\x05\x12\x17\n\x0f\x63heckpoint_path\x18\x02 \x01(\t\"x\n\x14StartTrainingRequest\x12\x13\n\x0btraining_id\x18\x01 \x01(\x05\x12\x0e\n\x06\x64\x65vice\x18\x02 \x01(\t\x12\x1d\n\x15train_until_sample_id\x18\x03 \x01(\t\x12\x1c\n\x14load_checkpoint_path\x18\x04 \x01(\t\"1\n\x15StartTrainingResponse\x12\x18\n\x10training_started\x18\x01 \x01(\x08\",\n\x15TrainingStatusRequest\x12\x13\n\x0btraining_id\x18\x01 \x01(\x05\"\xf2\x01\n\x16TrainingStatusResponse\x12\x12\n\nis_running\x18\x01 \x01(\x08\x12\x17\n\x0fstate_available\x18\x02 \x01(\x08\x12\x0f\n\x07\x62locked\x18\x03 \x01(\x08\x12\x16\n\texception\x18\x04 \x01(\tH\x00\x88\x01\x01\x12\x19\n\x0c\x62\x61tches_seen\x18\x05 \x01(\x05H\x01\x88\x01\x01\x12\x19\n\x0csamples_seen\x18\x06 \x01(\x05H\x02\x88\x01\x01\x12\x12\n\x05state\x18\x07 \x01(\x0cH\x03\x88\x01\x01\x42\x0c\n\n_exceptionB\x0f\n\r_batches_seenB\x0f\n\r_samples_seenB\x08\n\x06_state2\xf1\x02\n\rTrainerServer\x12W\n\x08register\x12#.trainer.RegisterTrainServerRequest\x1a$.trainer.RegisterTrainServerResponse\"\x00\x12Z\n\x11trainer_available\x12 .trainer.TrainerAvailableRequest\x1a!.trainer.TrainerAvailableResponse\"\x00\x12Q\n\x0estart_training\x12\x1d.trainer.StartTrainingRequest\x1a\x1e.trainer.StartTrainingResponse\"\x00\x12X\n\x13get_training_status\x12\x1e.trainer.TrainingStatusRequest\x1a\x1f.trainer.TrainingStatusResponse\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'trainer_server_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _REGISTERTRAINSERVERREQUEST._serialized_start=34
  _REGISTERTRAINSERVERREQUEST._serialized_end=431
  _JSONSTRING._serialized_start=433
  _JSONSTRING._serialized_end=460
  _REGISTERTRAINSERVERRESPONSE._serialized_start=462
  _REGISTERTRAINSERVERRESPONSE._serialized_end=508
  _DATA._serialized_start=510
  _DATA._serialized_end=561
  _TRAINERAVAILABLEREQUEST._serialized_start=563
  _TRAINERAVAILABLEREQUEST._serialized_end=588
  _TRAINERAVAILABLERESPONSE._serialized_start=590
  _TRAINERAVAILABLERESPONSE._serialized_end=635
  _CHECKPOINTINFO._serialized_start=637
  _CHECKPOINTINFO._serialized_end=707
  _STARTTRAININGREQUEST._serialized_start=709
  _STARTTRAININGREQUEST._serialized_end=829
  _STARTTRAININGRESPONSE._serialized_start=831
  _STARTTRAININGRESPONSE._serialized_end=880
  _TRAININGSTATUSREQUEST._serialized_start=882
  _TRAININGSTATUSREQUEST._serialized_end=926
  _TRAININGSTATUSRESPONSE._serialized_start=929
  _TRAININGSTATUSRESPONSE._serialized_end=1171
  _TRAINERSERVER._serialized_start=1174
  _TRAINERSERVER._serialized_end=1543
# @@protoc_insertion_point(module_scope)
