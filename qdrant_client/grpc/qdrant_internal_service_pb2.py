# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: qdrant_internal_service.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1dqdrant_internal_service.proto\x12\x06qdrant\"\x11\n\x0fHttpPortRequest\" \n\x10HttpPortResponse\x12\x0c\n\x04port\x18\x01 \x01(\x05\x32T\n\x0eQdrantInternal\x12\x42\n\x0bGetHttpPort\x12\x17.qdrant.HttpPortRequest\x1a\x18.qdrant.HttpPortResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'qdrant_internal_service_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _globals['_HTTPPORTREQUEST']._serialized_start=41
  _globals['_HTTPPORTREQUEST']._serialized_end=58
  _globals['_HTTPPORTRESPONSE']._serialized_start=60
  _globals['_HTTPPORTRESPONSE']._serialized_end=92
  _globals['_QDRANTINTERNAL']._serialized_start=94
  _globals['_QDRANTINTERNAL']._serialized_end=178
# @@protoc_insertion_point(module_scope)
