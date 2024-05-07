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




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0f\x65valuator.proto\x12\x0fmodyn.evaluator\"j\n\x0b\x44\x61tasetInfo\x12\x12\n\ndataset_id\x18\x01 \x01(\t\x12\x17\n\x0fnum_dataloaders\x18\x02 \x01(\x05\x12\x17\n\x0fstart_timestamp\x18\x03 \x01(\x03\x12\x15\n\rend_timestamp\x18\x04 \x01(\x03\"\x1d\n\x0cPythonString\x12\r\n\x05value\x18\x01 \x01(\t\"\x1b\n\nJsonString\x12\r\n\x05value\x18\x01 \x01(\t\"\x8f\x01\n\x13MetricConfiguration\x12\x0c\n\x04name\x18\x01 \x01(\t\x12+\n\x06\x63onfig\x18\x02 \x01(\x0b\x32\x1b.modyn.evaluator.JsonString\x12=\n\x16\x65valuation_transformer\x18\x03 \x01(\x0b\x32\x1d.modyn.evaluator.PythonString\"\x83\x03\n\x14\x45valuateModelRequest\x12\x10\n\x08model_id\x18\x01 \x01(\x05\x12\x32\n\x0c\x64\x61taset_info\x18\x02 \x01(\x0b\x32\x1c.modyn.evaluator.DatasetInfo\x12\x0e\n\x06\x64\x65vice\x18\x03 \x01(\t\x12\x12\n\nbatch_size\x18\x04 \x01(\x05\x12\x35\n\x07metrics\x18\x05 \x03(\x0b\x32$.modyn.evaluator.MetricConfiguration\x12\x16\n\x0etransform_list\x18\x06 \x03(\t\x12\x33\n\x0c\x62ytes_parser\x18\x07 \x01(\x0b\x32\x1d.modyn.evaluator.PythonString\x12\x38\n\x11label_transformer\x18\x08 \x01(\x0b\x32\x1d.modyn.evaluator.PythonString\x12\x35\n\ttokenizer\x18\n \x01(\x0b\x32\x1d.modyn.evaluator.PythonStringH\x00\x88\x01\x01\x42\x0c\n\n_tokenizer\"\xa5\x01\n\x15\x45valuateModelResponse\x12\x1a\n\x12\x65valuation_started\x18\x01 \x01(\x08\x12\x15\n\revaluation_id\x18\x02 \x01(\x05\x12\x14\n\x0c\x64\x61taset_size\x18\x03 \x01(\x03\x12\x43\n\x10not_start_reason\x18\x04 \x01(\x0e\x32).modyn.evaluator.EvaluationNotStartReason\"0\n\x17\x45valuationStatusRequest\x12\x15\n\revaluation_id\x18\x01 \x01(\x05\"\xe5\x01\n\x18\x45valuationStatusResponse\x12\r\n\x05valid\x18\x01 \x01(\x08\x12\x12\n\nis_running\x18\x02 \x01(\x08\x12\x17\n\x0fstate_available\x18\x03 \x01(\x08\x12\x0f\n\x07\x62locked\x18\x04 \x01(\x08\x12\x16\n\texception\x18\x05 \x01(\tH\x00\x88\x01\x01\x12\x19\n\x0c\x62\x61tches_seen\x18\x06 \x01(\x03H\x01\x88\x01\x01\x12\x19\n\x0csamples_seen\x18\x07 \x01(\x03H\x02\x88\x01\x01\x42\x0c\n\n_exceptionB\x0f\n\r_batches_seenB\x0f\n\r_samples_seen\"0\n\x0e\x45valuationData\x12\x0e\n\x06metric\x18\x01 \x01(\t\x12\x0e\n\x06result\x18\x02 \x01(\x02\"0\n\x17\x45valuationResultRequest\x12\x15\n\revaluation_id\x18\x01 \x01(\x05\"c\n\x18\x45valuationResultResponse\x12\r\n\x05valid\x18\x01 \x01(\x08\x12\x38\n\x0f\x65valuation_data\x18\x02 \x03(\x0b\x32\x1f.modyn.evaluator.EvaluationData*\xc8\x01\n\x18\x45valuationNotStartReason\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x1f\n\x1bMODEL_NOT_EXIST_IN_METADATA\x10\x01\x12\x18\n\x14MODEL_IMPORT_FAILURE\x10\x02\x12\x1e\n\x1aMODEL_NOT_EXIST_IN_STORAGE\x10\x03\x12\x15\n\x11\x44\x41TASET_NOT_FOUND\x10\x04\x12\x11\n\rEMPTY_DATASET\x10\x05\x12\x1a\n\x16\x44OWNLOAD_MODEL_FAILURE\x10\x06\x32\xce\x02\n\tEvaluator\x12\x61\n\x0e\x65valuate_model\x12%.modyn.evaluator.EvaluateModelRequest\x1a&.modyn.evaluator.EvaluateModelResponse\"\x00\x12n\n\x15get_evaluation_status\x12(.modyn.evaluator.EvaluationStatusRequest\x1a).modyn.evaluator.EvaluationStatusResponse\"\x00\x12n\n\x15get_evaluation_result\x12(.modyn.evaluator.EvaluationResultRequest\x1a).modyn.evaluator.EvaluationResultResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'evaluator_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_EVALUATIONNOTSTARTREASON']._serialized_start=1392
  _globals['_EVALUATIONNOTSTARTREASON']._serialized_end=1592
  _globals['_DATASETINFO']._serialized_start=36
  _globals['_DATASETINFO']._serialized_end=142
  _globals['_PYTHONSTRING']._serialized_start=144
  _globals['_PYTHONSTRING']._serialized_end=173
  _globals['_JSONSTRING']._serialized_start=175
  _globals['_JSONSTRING']._serialized_end=202
  _globals['_METRICCONFIGURATION']._serialized_start=205
  _globals['_METRICCONFIGURATION']._serialized_end=348
  _globals['_EVALUATEMODELREQUEST']._serialized_start=351
  _globals['_EVALUATEMODELREQUEST']._serialized_end=738
  _globals['_EVALUATEMODELRESPONSE']._serialized_start=741
  _globals['_EVALUATEMODELRESPONSE']._serialized_end=906
  _globals['_EVALUATIONSTATUSREQUEST']._serialized_start=908
  _globals['_EVALUATIONSTATUSREQUEST']._serialized_end=956
  _globals['_EVALUATIONSTATUSRESPONSE']._serialized_start=959
  _globals['_EVALUATIONSTATUSRESPONSE']._serialized_end=1188
  _globals['_EVALUATIONDATA']._serialized_start=1190
  _globals['_EVALUATIONDATA']._serialized_end=1238
  _globals['_EVALUATIONRESULTREQUEST']._serialized_start=1240
  _globals['_EVALUATIONRESULTREQUEST']._serialized_end=1288
  _globals['_EVALUATIONRESULTRESPONSE']._serialized_start=1290
  _globals['_EVALUATIONRESULTRESPONSE']._serialized_end=1389
  _globals['_EVALUATOR']._serialized_start=1595
  _globals['_EVALUATOR']._serialized_end=1929
# @@protoc_insertion_point(module_scope)
