# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: supervisor.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x10supervisor.proto\x12\nsupervisor\"\x1b\n\nJsonString\x12\r\n\x05value\x18\x01 \x01(\t\"\x90\x02\n\x14StartPipelineRequest\x12/\n\x0fpipeline_config\x18\x01 \x01(\x0b\x32\x16.supervisor.JsonString\x12\x16\n\x0e\x65val_directory\x18\x02 \x01(\t\x12\x1c\n\x0fstart_replay_at\x18\x03 \x01(\x05H\x00\x88\x01\x01\x12\x1b\n\x0estop_replay_at\x18\x04 \x01(\x05H\x01\x88\x01\x01\x12\x1d\n\x10maximum_triggers\x18\x05 \x01(\x05H\x02\x88\x01\x01\x12\x19\n\x11\x65valuation_matrix\x18\x06 \x01(\x08\x42\x12\n\x10_start_replay_atB\x11\n\x0f_stop_replay_atB\x13\n\x11_maximum_triggers\"M\n\x10PipelineResponse\x12\x13\n\x0bpipeline_id\x18\x01 \x01(\x05\x12\x16\n\texception\x18\x02 \x01(\tH\x00\x88\x01\x01\x42\x0c\n\n_exception\"/\n\x18GetPipelineStatusRequest\x12\x13\n\x0bpipeline_id\x18\x01 \x01(\x05\"1\n\x12PipelineStageIdMsg\x12\x0f\n\x07id_type\x18\x01 \x01(\t\x12\n\n\x02id\x18\x02 \x01(\x05\"%\n\x17PipelineStageDatasetMsg\x12\n\n\x02id\x18\x01 \x01(\t\"8\n PipelineStageCounterCreateParams\x12\x14\n\x0cnew_data_len\x18\x01 \x01(\x05\"6\n PipelineStageCounterUpdateParams\x12\x12\n\nbatch_size\x18\x01 \x01(\x05\"\xc1\x01\n\x17PipelineStageCounterMsg\x12\x0e\n\x06\x61\x63tion\x18\x01 \x01(\t\x12\x45\n\rcreate_params\x18\x02 \x01(\x0b\x32,.supervisor.PipelineStageCounterCreateParamsH\x00\x12\x45\n\rupdate_params\x18\x03 \x01(\x0b\x32,.supervisor.PipelineStageCounterUpdateParamsH\x00\x42\x08\n\x06params\"N\n\x14PipelineStageExitMsg\x12\x10\n\x08\x65xitcode\x18\x01 \x01(\x05\x12\x16\n\texception\x18\x02 \x01(\tH\x00\x88\x01\x01\x42\x0c\n\n_exception\"\xa4\x02\n\rPipelineStage\x12\r\n\x05stage\x18\x01 \x01(\t\x12\x10\n\x08msg_type\x18\x02 \x01(\t\x12\x0b\n\x03log\x18\x03 \x01(\x08\x12\x30\n\x06id_msg\x18\x04 \x01(\x0b\x32\x1e.supervisor.PipelineStageIdMsgH\x00\x12:\n\x0b\x64\x61taset_msg\x18\x05 \x01(\x0b\x32#.supervisor.PipelineStageDatasetMsgH\x00\x12:\n\x0b\x63ounter_msg\x18\x06 \x01(\x0b\x32#.supervisor.PipelineStageCounterMsgH\x00\x12\x34\n\x08\x65xit_msg\x18\x07 \x01(\x0b\x32 .supervisor.PipelineStageExitMsgH\x00\x42\x05\n\x03msg\"T\n!TrainingStatusCreateTrackerParams\x12\x15\n\rtotal_samples\x18\x01 \x01(\x05\x12\x18\n\x10status_bar_scale\x18\x02 \x01(\x05\"s\n#TrainingStatusProgressCounterParams\x12\x14\n\x0csamples_seen\x18\x01 \x01(\x05\x12!\n\x19\x64ownsampling_samples_seen\x18\x02 \x01(\x05\x12\x13\n\x0bis_training\x18\x03 \x01(\x08\"\xfb\x01\n\x0eTrainingStatus\x12\r\n\x05stage\x18\x01 \x01(\t\x12\x0e\n\x06\x61\x63tion\x18\x02 \x01(\t\x12\n\n\x02id\x18\x03 \x01(\x05\x12W\n\x1etraining_create_tracker_params\x18\x04 \x01(\x0b\x32-.supervisor.TrainingStatusCreateTrackerParamsH\x00\x12[\n training_progress_counter_params\x18\x05 \x01(\x0b\x32/.supervisor.TrainingStatusProgressCounterParamsH\x00\x42\x08\n\x06params\"I\n\x1d\x45valStatusCreateTrackerParams\x12\x12\n\ndataset_id\x18\x01 \x01(\t\x12\x14\n\x0c\x64\x61taset_size\x18\x02 \x01(\x05\"4\n\x1d\x45valStatusCreateCounterParams\x12\x13\n\x0btraining_id\x18\x01 \x01(\x05\"=\n\x1f\x45valStatusProgressCounterParams\x12\x1a\n\x12total_samples_seen\x18\x01 \x01(\x05\"Y\n\x1a\x45valStatusEndCounterParams\x12\r\n\x05\x65rror\x18\x01 \x01(\x08\x12\x1a\n\rexception_msg\x18\x02 \x01(\tH\x00\x88\x01\x01\x42\x10\n\x0e_exception_msg\"\x83\x03\n\nEvalStatus\x12\r\n\x05stage\x18\x01 \x01(\t\x12\x0e\n\x06\x61\x63tion\x18\x02 \x01(\t\x12\n\n\x02id\x18\x03 \x01(\x05\x12O\n\x1a\x65val_create_tracker_params\x18\x04 \x01(\x0b\x32).supervisor.EvalStatusCreateTrackerParamsH\x00\x12O\n\x1a\x65val_create_counter_params\x18\x05 \x01(\x0b\x32).supervisor.EvalStatusCreateCounterParamsH\x00\x12S\n\x1c\x65val_progress_counter_params\x18\x06 \x01(\x0b\x32+.supervisor.EvalStatusProgressCounterParamsH\x00\x12I\n\x17\x65val_end_counter_params\x18\x07 \x01(\x0b\x32&.supervisor.EvalStatusEndCounterParamsH\x00\x42\x08\n\x06params\"\xc0\x01\n\x19GetPipelineStatusResponse\x12\x0e\n\x06status\x18\x01 \x01(\t\x12\x31\n\x0epipeline_stage\x18\x02 \x03(\x0b\x32\x19.supervisor.PipelineStage\x12\x33\n\x0ftraining_status\x18\x03 \x03(\x0b\x32\x1a.supervisor.TrainingStatus\x12+\n\x0b\x65val_status\x18\x04 \x03(\x0b\x32\x16.supervisor.EvalStatus2\xc6\x01\n\nSupervisor\x12R\n\x0estart_pipeline\x12 .supervisor.StartPipelineRequest\x1a\x1c.supervisor.PipelineResponse\"\x00\x12\x64\n\x13get_pipeline_status\x12$.supervisor.GetPipelineStatusRequest\x1a%.supervisor.GetPipelineStatusResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'supervisor_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_JSONSTRING']._serialized_start=32
  _globals['_JSONSTRING']._serialized_end=59
  _globals['_STARTPIPELINEREQUEST']._serialized_start=62
  _globals['_STARTPIPELINEREQUEST']._serialized_end=334
  _globals['_PIPELINERESPONSE']._serialized_start=336
  _globals['_PIPELINERESPONSE']._serialized_end=413
  _globals['_GETPIPELINESTATUSREQUEST']._serialized_start=415
  _globals['_GETPIPELINESTATUSREQUEST']._serialized_end=462
  _globals['_PIPELINESTAGEIDMSG']._serialized_start=464
  _globals['_PIPELINESTAGEIDMSG']._serialized_end=513
  _globals['_PIPELINESTAGEDATASETMSG']._serialized_start=515
  _globals['_PIPELINESTAGEDATASETMSG']._serialized_end=552
  _globals['_PIPELINESTAGECOUNTERCREATEPARAMS']._serialized_start=554
  _globals['_PIPELINESTAGECOUNTERCREATEPARAMS']._serialized_end=610
  _globals['_PIPELINESTAGECOUNTERUPDATEPARAMS']._serialized_start=612
  _globals['_PIPELINESTAGECOUNTERUPDATEPARAMS']._serialized_end=666
  _globals['_PIPELINESTAGECOUNTERMSG']._serialized_start=669
  _globals['_PIPELINESTAGECOUNTERMSG']._serialized_end=862
  _globals['_PIPELINESTAGEEXITMSG']._serialized_start=864
  _globals['_PIPELINESTAGEEXITMSG']._serialized_end=942
  _globals['_PIPELINESTAGE']._serialized_start=945
  _globals['_PIPELINESTAGE']._serialized_end=1237
  _globals['_TRAININGSTATUSCREATETRACKERPARAMS']._serialized_start=1239
  _globals['_TRAININGSTATUSCREATETRACKERPARAMS']._serialized_end=1323
  _globals['_TRAININGSTATUSPROGRESSCOUNTERPARAMS']._serialized_start=1325
  _globals['_TRAININGSTATUSPROGRESSCOUNTERPARAMS']._serialized_end=1440
  _globals['_TRAININGSTATUS']._serialized_start=1443
  _globals['_TRAININGSTATUS']._serialized_end=1694
  _globals['_EVALSTATUSCREATETRACKERPARAMS']._serialized_start=1696
  _globals['_EVALSTATUSCREATETRACKERPARAMS']._serialized_end=1769
  _globals['_EVALSTATUSCREATECOUNTERPARAMS']._serialized_start=1771
  _globals['_EVALSTATUSCREATECOUNTERPARAMS']._serialized_end=1823
  _globals['_EVALSTATUSPROGRESSCOUNTERPARAMS']._serialized_start=1825
  _globals['_EVALSTATUSPROGRESSCOUNTERPARAMS']._serialized_end=1886
  _globals['_EVALSTATUSENDCOUNTERPARAMS']._serialized_start=1888
  _globals['_EVALSTATUSENDCOUNTERPARAMS']._serialized_end=1977
  _globals['_EVALSTATUS']._serialized_start=1980
  _globals['_EVALSTATUS']._serialized_end=2367
  _globals['_GETPIPELINESTATUSRESPONSE']._serialized_start=2370
  _globals['_GETPIPELINESTATUSRESPONSE']._serialized_end=2562
  _globals['_SUPERVISOR']._serialized_start=2565
  _globals['_SUPERVISOR']._serialized_end=2763
# @@protoc_insertion_point(module_scope)
