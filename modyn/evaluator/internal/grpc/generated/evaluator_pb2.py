# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: evaluator.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0f\x65valuator.proto\x12\x0fmodyn.evaluator\"t\n\x12\x45valuationInterval\x12\x1c\n\x0fstart_timestamp\x18\x01 \x01(\x03H\x00\x88\x01\x01\x12\x1a\n\rend_timestamp\x18\x02 \x01(\x03H\x01\x88\x01\x01\x42\x12\n\x10_start_timestampB\x10\n\x0e_end_timestamp\"}\n\x0b\x44\x61tasetInfo\x12\x12\n\ndataset_id\x18\x01 \x01(\t\x12\x17\n\x0fnum_dataloaders\x18\x02 \x01(\x05\x12\x41\n\x14\x65valuation_intervals\x18\x03 \x03(\x0b\x32#.modyn.evaluator.EvaluationInterval\"\x1d\n\x0cPythonString\x12\r\n\x05value\x18\x01 \x01(\t\"\x1b\n\nJsonString\x12\r\n\x05value\x18\x01 \x01(\t\"\xfa\x02\n\x14\x45valuateModelRequest\x12\x10\n\x08model_id\x18\x01 \x01(\x05\x12\x32\n\x0c\x64\x61taset_info\x18\x02 \x01(\x0b\x32\x1c.modyn.evaluator.DatasetInfo\x12\x0e\n\x06\x64\x65vice\x18\x03 \x01(\t\x12\x12\n\nbatch_size\x18\x04 \x01(\x05\x12,\n\x07metrics\x18\x05 \x03(\x0b\x32\x1b.modyn.evaluator.JsonString\x12\x16\n\x0etransform_list\x18\x06 \x03(\t\x12\x33\n\x0c\x62ytes_parser\x18\x07 \x01(\x0b\x32\x1d.modyn.evaluator.PythonString\x12\x38\n\x11label_transformer\x18\x08 \x01(\x0b\x32\x1d.modyn.evaluator.PythonString\x12\x35\n\ttokenizer\x18\t \x01(\x0b\x32\x1d.modyn.evaluator.PythonStringH\x00\x88\x01\x01\x42\x0c\n\n_tokenizer\"\xa9\x01\n\x15\x45valuateModelResponse\x12\x1a\n\x12\x65valuation_started\x18\x01 \x01(\x08\x12\x15\n\revaluation_id\x18\x02 \x01(\x05\x12\x15\n\rdataset_sizes\x18\x03 \x03(\x03\x12\x46\n\x14\x65val_aborted_reasons\x18\x04 \x03(\x0e\x32(.modyn.evaluator.EvaluationAbortedReason\"0\n\x17\x45valuationStatusRequest\x12\x15\n\revaluation_id\x18\x01 \x01(\x05\"c\n\x18\x45valuationStatusResponse\x12\r\n\x05valid\x18\x01 \x01(\x08\x12\x12\n\nis_running\x18\x02 \x01(\x08\x12\x16\n\texception\x18\x03 \x01(\tH\x00\x88\x01\x01\x42\x0c\n\n_exception\"4\n\x12SingleMetricResult\x12\x0e\n\x06metric\x18\x01 \x01(\t\x12\x0e\n\x06result\x18\x02 \x01(\x02\"l\n\x14SingleEvaluationData\x12\x16\n\x0einterval_index\x18\x01 \x01(\x05\x12<\n\x0f\x65valuation_data\x18\x02 \x03(\x0b\x32#.modyn.evaluator.SingleMetricResult\"0\n\x17\x45valuationResultRequest\x12\x15\n\revaluation_id\x18\x01 \x01(\x05\"2\n\x18\x45valuationCleanupRequest\x12\x16\n\x0e\x65valuation_ids\x18\x01 \x03(\x05\"l\n\x18\x45valuationResultResponse\x12\r\n\x05valid\x18\x01 \x01(\x08\x12\x41\n\x12\x65valuation_results\x18\x02 \x03(\x0b\x32%.modyn.evaluator.SingleEvaluationData\".\n\x19\x45valuationCleanupResponse\x12\x11\n\tsucceeded\x18\x01 \x03(\x05*\xc7\x01\n\x17\x45valuationAbortedReason\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x1f\n\x1bMODEL_NOT_EXIST_IN_METADATA\x10\x01\x12\x18\n\x14MODEL_IMPORT_FAILURE\x10\x02\x12\x1e\n\x1aMODEL_NOT_EXIST_IN_STORAGE\x10\x03\x12\x15\n\x11\x44\x41TASET_NOT_FOUND\x10\x04\x12\x11\n\rEMPTY_DATASET\x10\x05\x12\x1a\n\x16\x44OWNLOAD_MODEL_FAILURE\x10\x06\x32\xbe\x03\n\tEvaluator\x12\x61\n\x0e\x65valuate_model\x12%.modyn.evaluator.EvaluateModelRequest\x1a&.modyn.evaluator.EvaluateModelResponse\"\x00\x12n\n\x15get_evaluation_status\x12(.modyn.evaluator.EvaluationStatusRequest\x1a).modyn.evaluator.EvaluationStatusResponse\"\x00\x12n\n\x15get_evaluation_result\x12(.modyn.evaluator.EvaluationResultRequest\x1a).modyn.evaluator.EvaluationResultResponse\"\x00\x12n\n\x13\x63leanup_evaluations\x12).modyn.evaluator.EvaluationCleanupRequest\x1a*.modyn.evaluator.EvaluationCleanupResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'evaluator_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_EVALUATIONABORTEDREASON']._serialized_start=1470
  _globals['_EVALUATIONABORTEDREASON']._serialized_end=1669
  _globals['_EVALUATIONINTERVAL']._serialized_start=36
  _globals['_EVALUATIONINTERVAL']._serialized_end=152
  _globals['_DATASETINFO']._serialized_start=154
  _globals['_DATASETINFO']._serialized_end=279
  _globals['_PYTHONSTRING']._serialized_start=281
  _globals['_PYTHONSTRING']._serialized_end=310
  _globals['_JSONSTRING']._serialized_start=312
  _globals['_JSONSTRING']._serialized_end=339
  _globals['_EVALUATEMODELREQUEST']._serialized_start=342
  _globals['_EVALUATEMODELREQUEST']._serialized_end=720
  _globals['_EVALUATEMODELRESPONSE']._serialized_start=723
  _globals['_EVALUATEMODELRESPONSE']._serialized_end=892
  _globals['_EVALUATIONSTATUSREQUEST']._serialized_start=894
  _globals['_EVALUATIONSTATUSREQUEST']._serialized_end=942
  _globals['_EVALUATIONSTATUSRESPONSE']._serialized_start=944
  _globals['_EVALUATIONSTATUSRESPONSE']._serialized_end=1043
  _globals['_SINGLEMETRICRESULT']._serialized_start=1045
  _globals['_SINGLEMETRICRESULT']._serialized_end=1097
  _globals['_SINGLEEVALUATIONDATA']._serialized_start=1099
  _globals['_SINGLEEVALUATIONDATA']._serialized_end=1207
  _globals['_EVALUATIONRESULTREQUEST']._serialized_start=1209
  _globals['_EVALUATIONRESULTREQUEST']._serialized_end=1257
  _globals['_EVALUATIONCLEANUPREQUEST']._serialized_start=1259
  _globals['_EVALUATIONCLEANUPREQUEST']._serialized_end=1309
  _globals['_EVALUATIONRESULTRESPONSE']._serialized_start=1311
  _globals['_EVALUATIONRESULTRESPONSE']._serialized_end=1419
  _globals['_EVALUATIONCLEANUPRESPONSE']._serialized_start=1421
  _globals['_EVALUATIONCLEANUPRESPONSE']._serialized_end=1467
  _globals['_EVALUATOR']._serialized_start=1672
  _globals['_EVALUATOR']._serialized_end=2118
# @@protoc_insertion_point(module_scope)
