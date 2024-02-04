from .points_pb2 import *
from .collections_pb2 import *
from .snapshots_service_pb2 import *
from .json_with_int_pb2 import *
from .collections_service_pb2_grpc import *
from .points_service_pb2_grpc import *
from .snapshots_service_pb2_grpc import *


# Compression Enum Implementation

import enum
import grpc

@enum.unique
class Compression(enum.IntEnum):
    """Defines a custom compression enum tailored for Qdrant RUST Server RPCs.

    Qdrant RUST Server only supports specific compression algorithms for RPC communication.
    This enum mirrors the supported compression options, excluding 'Deflate,' which is not
    supported by the server. Use these compression options when specifying the desired
    compression method for interactions with the Qdrant RUST Server.

    Attributes:
     NoCompression: Do not use any compression algorithm.
     Gzip: Use the "Gzip" compression algorithm.
    """

    NoCompression = grpc.Compression.NoCompression
    Gzip = grpc.Compression.Gzip
