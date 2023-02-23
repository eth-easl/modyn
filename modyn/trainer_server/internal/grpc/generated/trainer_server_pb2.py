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




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14trainer_server.proto\x12\x07trainer\"\x1b\n\nJsonString\x12\r\n\x05value\x18\x01 \x01(\t\"\x1d\n\x0cPythonString\x12\r\n\x05value\x18\x01 \x01(\t\"3\n\x04\x44\x61ta\x12\x12\n\ndataset_id\x18\x01 \x01(\t\x12\x17\n\x0fnum_dataloaders\x18\x02 \x01(\x05\"\x19\n\x17TrainerAvailableRequest\"-\n\x18TrainerAvailableResponse\x12\x11\n\tavailable\x18\x01 \x01(\x08\"F\n\x0e\x43heckpointInfo\x12\x1b\n\x13\x63heckpoint_interval\x18\x01 \x01(\x05\x12\x17\n\x0f\x63heckpoint_path\x18\x02 \x01(\t\"\xfc\x04\n\x14StartTrainingRequest\x12\x13\n\x0bpipeline_id\x18\x01 \x01(\x05\x12\x12\n\ntrigger_id\x18\x02 \x01(\x05\x12\x0e\n\x06\x64\x65vice\x18\x03 \x01(\t\x12\x10\n\x08model_id\x18\x04 \x01(\t\x12\x30\n\x13model_configuration\x18\x05 \x01(\x0b\x32\x13.trainer.JsonString\x12\x1c\n\x14use_pretrained_model\x18\x06 \x01(\x08\x12\x1c\n\x14load_optimizer_state\x18\x07 \x01(\x08\x12\x18\n\x10pretrained_model\x18\x08 \x01(\x0c\x12\x12\n\nbatch_size\x18\t \x01(\x05\x12;\n\x1etorch_optimizers_configuration\x18\n \x01(\x0b\x32\x13.trainer.JsonString\x12\x17\n\x0ftorch_criterion\x18\x0b \x01(\t\x12\x31\n\x14\x63riterion_parameters\x18\x0c \x01(\x0b\x32\x13.trainer.JsonString\x12 \n\tdata_info\x18\r \x01(\x0b\x32\r.trainer.Data\x12\x30\n\x0f\x63heckpoint_info\x18\x0e \x01(\x0b\x32\x17.trainer.CheckpointInfo\x12+\n\x0c\x62ytes_parser\x18\x0f \x01(\x0b\x32\x15.trainer.PythonString\x12\x16\n\x0etransform_list\x18\x10 \x03(\t\x12)\n\x0clr_scheduler\x18\x11 \x01(\x0b\x32\x13.trainer.JsonString\x12\x30\n\x11label_transformer\x18\x12 \x01(\x0b\x32\x15.trainer.PythonString\"F\n\x15StartTrainingResponse\x12\x18\n\x10training_started\x18\x01 \x01(\x08\x12\x13\n\x0btraining_id\x18\x02 \x01(\x05\",\n\x15TrainingStatusRequest\x12\x13\n\x0btraining_id\x18\x01 \x01(\x05\"\xe3\x01\n\x16TrainingStatusResponse\x12\r\n\x05valid\x18\x01 \x01(\x08\x12\x12\n\nis_running\x18\x02 \x01(\x08\x12\x17\n\x0fstate_available\x18\x03 \x01(\x08\x12\x0f\n\x07\x62locked\x18\x04 \x01(\x08\x12\x16\n\texception\x18\x05 \x01(\tH\x00\x88\x01\x01\x12\x19\n\x0c\x62\x61tches_seen\x18\x06 \x01(\x05H\x01\x88\x01\x01\x12\x19\n\x0csamples_seen\x18\x07 \x01(\x05H\x02\x88\x01\x01\x42\x0c\n\n_exceptionB\x0f\n\r_batches_seenB\x0f\n\r_samples_seen\"+\n\x14GetFinalModelRequest\x12\x13\n\x0btraining_id\x18\x01 \x01(\x05\";\n\x15GetFinalModelResponse\x12\x13\n\x0bvalid_state\x18\x01 \x01(\x08\x12\r\n\x05state\x18\x02 \x01(\x0c\",\n\x15GetLatestModelRequest\x12\x13\n\x0btraining_id\x18\x01 \x01(\x05\"<\n\x16GetLatestModelResponse\x12\x13\n\x0bvalid_state\x18\x01 \x01(\x08\x12\r\n\x05state\x18\x02 \x01(\x0c\x32\xc3\x03\n\rTrainerServer\x12Z\n\x11trainer_available\x12 .trainer.TrainerAvailableRequest\x1a!.trainer.TrainerAvailableResponse\"\x00\x12Q\n\x0estart_training\x12\x1d.trainer.StartTrainingRequest\x1a\x1e.trainer.StartTrainingResponse\"\x00\x12X\n\x13get_training_status\x12\x1e.trainer.TrainingStatusRequest\x1a\x1f.trainer.TrainingStatusResponse\"\x00\x12R\n\x0fget_final_model\x12\x1d.trainer.GetFinalModelRequest\x1a\x1e.trainer.GetFinalModelResponse\"\x00\x12U\n\x10get_latest_model\x12\x1e.trainer.GetLatestModelRequest\x1a\x1f.trainer.GetLatestModelResponse\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'trainer_server_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _JSONSTRING._serialized_start=33
  _JSONSTRING._serialized_end=60
  _PYTHONSTRING._serialized_start=62
  _PYTHONSTRING._serialized_end=91
  _DATA._serialized_start=93
  _DATA._serialized_end=144
  _TRAINERAVAILABLEREQUEST._serialized_start=146
  _TRAINERAVAILABLEREQUEST._serialized_end=171
  _TRAINERAVAILABLERESPONSE._serialized_start=173
  _TRAINERAVAILABLERESPONSE._serialized_end=218
  _CHECKPOINTINFO._serialized_start=220
  _CHECKPOINTINFO._serialized_end=290
  _STARTTRAININGREQUEST._serialized_start=293
  _STARTTRAININGREQUEST._serialized_end=929
  _STARTTRAININGRESPONSE._serialized_start=931
  _STARTTRAININGRESPONSE._serialized_end=1001
  _TRAININGSTATUSREQUEST._serialized_start=1003
  _TRAININGSTATUSREQUEST._serialized_end=1047
  _TRAININGSTATUSRESPONSE._serialized_start=1050
  _TRAININGSTATUSRESPONSE._serialized_end=1277
  _GETFINALMODELREQUEST._serialized_start=1279
  _GETFINALMODELREQUEST._serialized_end=1322
  _GETFINALMODELRESPONSE._serialized_start=1324
  _GETFINALMODELRESPONSE._serialized_end=1383
  _GETLATESTMODELREQUEST._serialized_start=1385
  _GETLATESTMODELREQUEST._serialized_end=1429
  _GETLATESTMODELRESPONSE._serialized_start=1431
  _GETLATESTMODELRESPONSE._serialized_end=1491
  _TRAINERSERVER._serialized_start=1494
  _TRAINERSERVER._serialized_end=1945
# @@protoc_insertion_point(module_scope)
