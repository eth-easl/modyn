# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: evaluator.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0f\x65valuator.proto\x12\x0fmodyn.evaluator\"j\n\x0b\x44\x61tasetInfo\x12\x12\n\ndataset_id\x18\x01 \x01(\t\x12\x17\n\x0fnum_dataloaders\x18\x02 \x01(\x05\x12\x17\n\x0fstart_timestamp\x18\x03 \x01(\x03\x12\x15\n\rend_timestamp\x18\x04 \x01(\x03\"\x1d\n\x0cPythonString\x12\r\n\x05value\x18\x01 \x01(\t\"\x1b\n\nJsonString\x12\r\n\x05value\x18\x01 \x01(\t\"\x8f\x01\n\x13MetricConfiguration\x12\x0c\n\x04name\x18\x01 \x01(\t\x12+\n\x06\x63onfig\x18\x02 \x01(\x0b\x32\x1b.modyn.evaluator.JsonString\x12=\n\x16\x65valuation_transformer\x18\x03 \x01(\x0b\x32\x1d.modyn.evaluator.PythonString\"\xbe\x02\n\x14\x45valuateModelRequest\x12\x10\n\x08model_id\x18\x01 \x01(\x05\x12\x32\n\x0c\x64\x61taset_info\x18\x02 \x01(\x0b\x32\x1c.modyn.evaluator.DatasetInfo\x12\x0e\n\x06\x64\x65vice\x18\x03 \x01(\t\x12\x12\n\nbatch_size\x18\x04 \x01(\x05\x12\x35\n\x07metrics\x18\x05 \x03(\x0b\x32$.modyn.evaluator.MetricConfiguration\x12\x16\n\x0etransform_list\x18\x06 \x03(\t\x12\x33\n\x0c\x62ytes_parser\x18\x07 \x01(\x0b\x32\x1d.modyn.evaluator.PythonString\x12\x38\n\x11label_transformer\x18\x08 \x01(\x0b\x32\x1d.modyn.evaluator.PythonString\"`\n\x15\x45valuateModelResponse\x12\x1a\n\x12\x65valuation_started\x18\x01 \x01(\x08\x12\x15\n\revaluation_id\x18\x02 \x01(\x05\x12\x14\n\x0c\x64\x61taset_size\x18\x03 \x01(\x03\"0\n\x17\x45valuationStatusRequest\x12\x15\n\revaluation_id\x18\x01 \x01(\x05\"\xe5\x01\n\x18\x45valuationStatusResponse\x12\r\n\x05valid\x18\x01 \x01(\x08\x12\x12\n\nis_running\x18\x02 \x01(\x08\x12\x17\n\x0fstate_available\x18\x03 \x01(\x08\x12\x0f\n\x07\x62locked\x18\x04 \x01(\x08\x12\x16\n\texception\x18\x05 \x01(\tH\x00\x88\x01\x01\x12\x19\n\x0c\x62\x61tches_seen\x18\x06 \x01(\x03H\x01\x88\x01\x01\x12\x19\n\x0csamples_seen\x18\x07 \x01(\x03H\x02\x88\x01\x01\x42\x0c\n\n_exceptionB\x0f\n\r_batches_seenB\x0f\n\r_samples_seen\"0\n\x0e\x45valuationData\x12\x0e\n\x06metric\x18\x01 \x01(\t\x12\x0e\n\x06result\x18\x02 \x01(\x02\"0\n\x17\x45valuationResultRequest\x12\x15\n\revaluation_id\x18\x01 \x01(\x05\"c\n\x18\x45valuationResultResponse\x12\r\n\x05valid\x18\x01 \x01(\x08\x12\x38\n\x0f\x65valuation_data\x18\x02 \x03(\x0b\x32\x1f.modyn.evaluator.EvaluationData2\xce\x02\n\tEvaluator\x12\x61\n\x0e\x65valuate_model\x12%.modyn.evaluator.EvaluateModelRequest\x1a&.modyn.evaluator.EvaluateModelResponse\"\x00\x12n\n\x15get_evaluation_status\x12(.modyn.evaluator.EvaluationStatusRequest\x1a).modyn.evaluator.EvaluationStatusResponse\"\x00\x12n\n\x15get_evaluation_result\x12(.modyn.evaluator.EvaluationResultRequest\x1a).modyn.evaluator.EvaluationResultResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'evaluator_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_DATASETINFO']._serialized_start=36
  _globals['_DATASETINFO']._serialized_end=142
  _globals['_PYTHONSTRING']._serialized_start=144
  _globals['_PYTHONSTRING']._serialized_end=173
  _globals['_JSONSTRING']._serialized_start=175
  _globals['_JSONSTRING']._serialized_end=202
  _globals['_METRICCONFIGURATION']._serialized_start=205
  _globals['_METRICCONFIGURATION']._serialized_end=348
  _globals['_EVALUATEMODELREQUEST']._serialized_start=351
  _globals['_EVALUATEMODELREQUEST']._serialized_end=669
  _globals['_EVALUATEMODELRESPONSE']._serialized_start=671
  _globals['_EVALUATEMODELRESPONSE']._serialized_end=767
  _globals['_EVALUATIONSTATUSREQUEST']._serialized_start=769
  _globals['_EVALUATIONSTATUSREQUEST']._serialized_end=817
  _globals['_EVALUATIONSTATUSRESPONSE']._serialized_start=820
  _globals['_EVALUATIONSTATUSRESPONSE']._serialized_end=1049
  _globals['_EVALUATIONDATA']._serialized_start=1051
  _globals['_EVALUATIONDATA']._serialized_end=1099
  _globals['_EVALUATIONRESULTREQUEST']._serialized_start=1101
  _globals['_EVALUATIONRESULTREQUEST']._serialized_end=1149
  _globals['_EVALUATIONRESULTRESPONSE']._serialized_start=1151
  _globals['_EVALUATIONRESULTRESPONSE']._serialized_end=1250
  _globals['_EVALUATOR']._serialized_start=1253
  _globals['_EVALUATOR']._serialized_end=1587
# @@protoc_insertion_point(module_scope)
