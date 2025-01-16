CACHE_STR_PATH = {
    "AbortShardTransfer": [],
    "AbortTransferOperation": [],
    "Batch": ["vectors"],
    "BinaryQuantization": [],
    "BinaryQuantizationConfig": [],
    "BoolIndexParams": [],
    "ChangeAliasesOperation": [],
    "ClearPayloadOperation": [],
    "CollectionParamsDiff": [],
    "ContextExamplePair": [],
    "ContextPair": ["negative", "positive"],
    "ContextQuery": ["context.negative", "context.positive"],
    "CountRequest": [],
    "CreateAlias": [],
    "CreateAliasOperation": [],
    "CreateCollection": [],
    "CreateFieldIndex": [],
    "CreateShardingKey": [],
    "CreateShardingKeyOperation": [],
    "DatetimeIndexParams": [],
    "DatetimeRange": [],
    "DeleteAlias": [],
    "DeleteAliasOperation": [],
    "DeleteOperation": [],
    "DeletePayload": [],
    "DeletePayloadOperation": [],
    "DeleteVectors": [],
    "DeleteVectorsOperation": [],
    "DiscoverInput": ["context.negative", "context.positive", "target"],
    "DiscoverQuery": ["discover.context.negative", "discover.context.positive", "discover.target"],
    "DiscoverRequest": [],
    "DiscoverRequestBatch": [],
    "Document": [""],
    "DropReplicaOperation": [],
    "DropShardingKey": [],
    "DropShardingKeyOperation": [],
    "FacetRequest": [],
    "FieldCondition": [],
    "Filter": [],
    "FilterSelector": [],
    "FloatIndexParams": [],
    "FusionQuery": [],
    "GeoBoundingBox": [],
    "GeoIndexParams": [],
    "GeoLineString": [],
    "GeoPoint": [],
    "GeoPolygon": [],
    "GeoRadius": [],
    "HasIdCondition": [],
    "HnswConfigDiff": [],
    "InitFrom": [],
    "IntegerIndexParams": [],
    "IsEmptyCondition": [],
    "IsNullCondition": [],
    "KeywordIndexParams": [],
    "LocksOption": [],
    "LookupLocation": [],
    "MatchAny": [],
    "MatchExcept": [],
    "MatchText": [],
    "MatchValue": [],
    "MinShould": [],
    "MoveShard": [],
    "MoveShardOperation": [],
    "MultiVectorConfig": [],
    "NamedSparseVector": [],
    "NamedVector": [],
    "NearestQuery": ["nearest"],
    "Nested": [],
    "NestedCondition": [],
    "OptimizersConfigDiff": [],
    "OrderBy": [],
    "OrderByQuery": [],
    "OverwritePayloadOperation": [],
    "PayloadField": [],
    "PayloadSelectorExclude": [],
    "PayloadSelectorInclude": [],
    "PointIdsList": [],
    "PointRequest": [],
    "PointStruct": ["vector"],
    "PointVectors": ["vector"],
    "PointsBatch": ["batch.vectors"],
    "PointsList": ["points.vector"],
    "Prefetch": [
        "prefetch.query",
        "prefetch.query.context.negative",
        "prefetch.query.context.positive",
        "prefetch.query.discover.context.negative",
        "prefetch.query.discover.context.positive",
        "prefetch.query.discover.target",
        "prefetch.query.nearest",
        "prefetch.query.recommend.negative",
        "prefetch.query.recommend.positive",
        "query",
        "query.context.negative",
        "query.context.positive",
        "query.discover.context.negative",
        "query.discover.context.positive",
        "query.discover.target",
        "query.nearest",
        "query.recommend.negative",
        "query.recommend.positive",
    ],
    "ProductQuantization": [],
    "ProductQuantizationConfig": [],
    "QuantizationSearchParams": [],
    "QueryGroupsRequest": [
        "prefetch.query",
        "prefetch.query.context.negative",
        "prefetch.query.context.positive",
        "prefetch.query.discover.context.negative",
        "prefetch.query.discover.context.positive",
        "prefetch.query.discover.target",
        "prefetch.query.nearest",
        "prefetch.query.recommend.negative",
        "prefetch.query.recommend.positive",
        "query",
        "query.context.negative",
        "query.context.positive",
        "query.discover.context.negative",
        "query.discover.context.positive",
        "query.discover.target",
        "query.nearest",
        "query.recommend.negative",
        "query.recommend.positive",
    ],
    "QueryRequest": [
        "prefetch.query",
        "prefetch.query.context.negative",
        "prefetch.query.context.positive",
        "prefetch.query.discover.context.negative",
        "prefetch.query.discover.context.positive",
        "prefetch.query.discover.target",
        "prefetch.query.nearest",
        "prefetch.query.recommend.negative",
        "prefetch.query.recommend.positive",
        "query",
        "query.context.negative",
        "query.context.positive",
        "query.discover.context.negative",
        "query.discover.context.positive",
        "query.discover.target",
        "query.nearest",
        "query.recommend.negative",
        "query.recommend.positive",
    ],
    "QueryRequestBatch": [
        "searches.prefetch.query",
        "searches.prefetch.query.context.negative",
        "searches.prefetch.query.context.positive",
        "searches.prefetch.query.discover.context.negative",
        "searches.prefetch.query.discover.context.positive",
        "searches.prefetch.query.discover.target",
        "searches.prefetch.query.nearest",
        "searches.prefetch.query.recommend.negative",
        "searches.prefetch.query.recommend.positive",
        "searches.query",
        "searches.query.context.negative",
        "searches.query.context.positive",
        "searches.query.discover.context.negative",
        "searches.query.discover.context.positive",
        "searches.query.discover.target",
        "searches.query.nearest",
        "searches.query.recommend.negative",
        "searches.query.recommend.positive",
    ],
    "QueryResponse": ["document"],
    "Range": [],
    "RecommendGroupsRequest": [],
    "RecommendInput": ["negative", "positive"],
    "RecommendQuery": ["recommend.negative", "recommend.positive"],
    "RecommendRequest": [],
    "RecommendRequestBatch": [],
    "RenameAlias": [],
    "RenameAliasOperation": [],
    "Replica": [],
    "ReplicateShard": [],
    "ReplicateShardOperation": [],
    "RestartTransfer": [],
    "RestartTransferOperation": [],
    "SampleQuery": [],
    "ScalarQuantization": [],
    "ScalarQuantizationConfig": [],
    "ScrollRequest": [],
    "SearchGroupsRequest": [],
    "SearchMatrixRequest": [],
    "SearchParams": [],
    "SearchRequest": [],
    "SearchRequestBatch": [],
    "SetPayload": [],
    "SetPayloadOperation": [],
    "ShardSnapshotRecover": [],
    "SnapshotRecover": [],
    "SparseIndexParams": [],
    "SparseVector": [],
    "SparseVectorParams": [],
    "TextIndexParams": [],
    "UpdateCollection": [],
    "UpdateOperations": [
        "operations.update_vectors.points.vector",
        "operations.upsert.batch.vectors",
        "operations.upsert.points.vector",
    ],
    "UpdateVectors": ["points.vector"],
    "UpdateVectorsOperation": ["update_vectors.points.vector"],
    "UpsertOperation": ["upsert.batch.vectors", "upsert.points.vector"],
    "UuidIndexParams": [],
    "ValuesCount": [],
    "VectorParams": [],
    "VectorParamsDiff": [],
    "WalConfigDiff": [],
    "WithLookup": [],
    "Image": [],
    "InferenceObject": [],
    "StrictModeConfig": [],
    "HasVectorCondition": [],
    "AbortReshardingOperation": [],
    "StartResharding": [],
    "StartReshardingOperation": [],
}
DEFS = {
    "AbortShardTransfer": {
        "additionalProperties": False,
        "properties": {
            "shard_id": {"description": "", "title": "Shard Id", "type": "integer"},
            "to_peer_id": {"description": "", "title": "To Peer Id", "type": "integer"},
            "from_peer_id": {"description": "", "title": "From Peer Id", "type": "integer"},
        },
        "required": ["shard_id", "to_peer_id", "from_peer_id"],
        "title": "AbortShardTransfer",
        "type": "object",
    },
    "Document": {
        "additionalProperties": False,
        "description": "WARN: Work-in-progress, unimplemented  Text document for embedding. Requires inference infrastructure, unimplemented.",
        "properties": {
            "text": {
                "description": "Text document to be embedded by FastEmbed or Cloud inference server",
                "title": "Text",
                "type": "string",
            },
            "model": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "description": "Model name to be used for embedding computation",
                "title": "Model",
            },
        },
        "required": ["text"],
        "title": "Document",
        "type": "object",
    },
    "SparseVector": {
        "additionalProperties": False,
        "description": "Sparse vector structure",
        "properties": {
            "indices": {
                "description": "Indices must be unique",
                "items": {"type": "integer"},
                "title": "Indices",
                "type": "array",
            },
            "values": {
                "description": "Values and indices must be the same length",
                "items": {"type": "number"},
                "title": "Values",
                "type": "array",
            },
        },
        "required": ["indices", "values"],
        "title": "SparseVector",
        "type": "object",
    },
    "BinaryQuantizationConfig": {
        "additionalProperties": False,
        "properties": {
            "always_ram": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "",
                "title": "Always Ram",
            }
        },
        "title": "BinaryQuantizationConfig",
        "type": "object",
    },
    "BoolIndexType": {
        "const": "bool",
        "enum": ["bool"],
        "title": "BoolIndexType",
        "type": "string",
    },
    "CreateAlias": {
        "additionalProperties": False,
        "description": "Create alternative name for a collection. Collection will be available under both names for search, retrieve,",
        "properties": {
            "collection_name": {
                "description": "Create alternative name for a collection. Collection will be available under both names for search, retrieve,",
                "title": "Collection Name",
                "type": "string",
            },
            "alias_name": {
                "description": "Create alternative name for a collection. Collection will be available under both names for search, retrieve,",
                "title": "Alias Name",
                "type": "string",
            },
        },
        "required": ["collection_name", "alias_name"],
        "title": "CreateAlias",
        "type": "object",
    },
    "CreateAliasOperation": {
        "additionalProperties": False,
        "properties": {
            "create_alias": {"allOf": [{"$ref": "#/$defs/CreateAlias"}], "description": ""}
        },
        "required": ["create_alias"],
        "title": "CreateAliasOperation",
        "type": "object",
    },
    "DeleteAlias": {
        "additionalProperties": False,
        "description": "Delete alias if exists",
        "properties": {
            "alias_name": {
                "description": "Delete alias if exists",
                "title": "Alias Name",
                "type": "string",
            }
        },
        "required": ["alias_name"],
        "title": "DeleteAlias",
        "type": "object",
    },
    "DeleteAliasOperation": {
        "additionalProperties": False,
        "description": "Delete alias if exists",
        "properties": {"delete_alias": {"$ref": "#/$defs/DeleteAlias"}},
        "required": ["delete_alias"],
        "title": "DeleteAliasOperation",
        "type": "object",
    },
    "RenameAlias": {
        "additionalProperties": False,
        "description": "Change alias to a new one",
        "properties": {
            "old_alias_name": {
                "description": "Change alias to a new one",
                "title": "Old Alias Name",
                "type": "string",
            },
            "new_alias_name": {
                "description": "Change alias to a new one",
                "title": "New Alias Name",
                "type": "string",
            },
        },
        "required": ["old_alias_name", "new_alias_name"],
        "title": "RenameAlias",
        "type": "object",
    },
    "RenameAliasOperation": {
        "additionalProperties": False,
        "description": "Change alias to a new one",
        "properties": {"rename_alias": {"$ref": "#/$defs/RenameAlias"}},
        "required": ["rename_alias"],
        "title": "RenameAliasOperation",
        "type": "object",
    },
    "DatetimeRange": {
        "additionalProperties": False,
        "description": "Range filter request",
        "properties": {
            "lt": {
                "anyOf": [
                    {"format": "date-time", "type": "string"},
                    {"format": "date", "type": "string"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "point.key &lt; range.lt",
                "title": "Lt",
            },
            "gt": {
                "anyOf": [
                    {"format": "date-time", "type": "string"},
                    {"format": "date", "type": "string"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "point.key &gt; range.gt",
                "title": "Gt",
            },
            "gte": {
                "anyOf": [
                    {"format": "date-time", "type": "string"},
                    {"format": "date", "type": "string"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "point.key &gt;= range.gte",
                "title": "Gte",
            },
            "lte": {
                "anyOf": [
                    {"format": "date-time", "type": "string"},
                    {"format": "date", "type": "string"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "point.key &lt;= range.lte",
                "title": "Lte",
            },
        },
        "title": "DatetimeRange",
        "type": "object",
    },
    "FieldCondition": {
        "additionalProperties": False,
        "description": "All possible payload filtering conditions",
        "properties": {
            "key": {"description": "Payload key", "title": "Key", "type": "string"},
            "match": {
                "anyOf": [
                    {"$ref": "#/$defs/MatchValue"},
                    {"$ref": "#/$defs/MatchText"},
                    {"$ref": "#/$defs/MatchAny"},
                    {"$ref": "#/$defs/MatchExcept"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Check if point has field with a given value",
                "title": "Match",
            },
            "range": {
                "anyOf": [
                    {"$ref": "#/$defs/Range"},
                    {"$ref": "#/$defs/DatetimeRange"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Check if points value lies in a given range",
                "title": "Range",
            },
            "geo_bounding_box": {
                "anyOf": [{"$ref": "#/$defs/GeoBoundingBox"}, {"type": "null"}],
                "default": None,
                "description": "Check if points geo location lies in a given area",
            },
            "geo_radius": {
                "anyOf": [{"$ref": "#/$defs/GeoRadius"}, {"type": "null"}],
                "default": None,
                "description": "Check if geo point is within a given radius",
            },
            "geo_polygon": {
                "anyOf": [{"$ref": "#/$defs/GeoPolygon"}, {"type": "null"}],
                "default": None,
                "description": "Check if geo point is within a given polygon",
            },
            "values_count": {
                "anyOf": [{"$ref": "#/$defs/ValuesCount"}, {"type": "null"}],
                "default": None,
                "description": "Check number of values of the field",
            },
        },
        "required": ["key"],
        "title": "FieldCondition",
        "type": "object",
    },
    "Filter": {
        "additionalProperties": False,
        "properties": {
            "should": {
                "anyOf": [
                    {
                        "items": {
                            "anyOf": [
                                {"$ref": "#/$defs/FieldCondition"},
                                {"$ref": "#/$defs/IsEmptyCondition"},
                                {"$ref": "#/$defs/IsNullCondition"},
                                {"$ref": "#/$defs/HasIdCondition"},
                                {"$ref": "#/$defs/NestedCondition"},
                                {"$ref": "#/$defs/Filter"},
                            ]
                        },
                        "type": "array",
                    },
                    {"$ref": "#/$defs/FieldCondition"},
                    {"$ref": "#/$defs/IsEmptyCondition"},
                    {"$ref": "#/$defs/IsNullCondition"},
                    {"$ref": "#/$defs/HasIdCondition"},
                    {"$ref": "#/$defs/NestedCondition"},
                    {"$ref": "#/$defs/Filter"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "At least one of those conditions should match",
                "title": "Should",
            },
            "min_should": {
                "anyOf": [{"$ref": "#/$defs/MinShould"}, {"type": "null"}],
                "default": None,
                "description": "At least minimum amount of given conditions should match",
            },
            "must": {
                "anyOf": [
                    {
                        "items": {
                            "anyOf": [
                                {"$ref": "#/$defs/FieldCondition"},
                                {"$ref": "#/$defs/IsEmptyCondition"},
                                {"$ref": "#/$defs/IsNullCondition"},
                                {"$ref": "#/$defs/HasIdCondition"},
                                {"$ref": "#/$defs/NestedCondition"},
                                {"$ref": "#/$defs/Filter"},
                            ]
                        },
                        "type": "array",
                    },
                    {"$ref": "#/$defs/FieldCondition"},
                    {"$ref": "#/$defs/IsEmptyCondition"},
                    {"$ref": "#/$defs/IsNullCondition"},
                    {"$ref": "#/$defs/HasIdCondition"},
                    {"$ref": "#/$defs/NestedCondition"},
                    {"$ref": "#/$defs/Filter"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "All conditions must match",
                "title": "Must",
            },
            "must_not": {
                "anyOf": [
                    {
                        "items": {
                            "anyOf": [
                                {"$ref": "#/$defs/FieldCondition"},
                                {"$ref": "#/$defs/IsEmptyCondition"},
                                {"$ref": "#/$defs/IsNullCondition"},
                                {"$ref": "#/$defs/HasIdCondition"},
                                {"$ref": "#/$defs/NestedCondition"},
                                {"$ref": "#/$defs/Filter"},
                            ]
                        },
                        "type": "array",
                    },
                    {"$ref": "#/$defs/FieldCondition"},
                    {"$ref": "#/$defs/IsEmptyCondition"},
                    {"$ref": "#/$defs/IsNullCondition"},
                    {"$ref": "#/$defs/HasIdCondition"},
                    {"$ref": "#/$defs/NestedCondition"},
                    {"$ref": "#/$defs/Filter"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "All conditions must NOT match",
                "title": "Must Not",
            },
        },
        "title": "Filter",
        "type": "object",
    },
    "FilterSelector": {
        "additionalProperties": False,
        "properties": {
            "filter": {"allOf": [{"$ref": "#/$defs/Filter"}], "description": ""},
            "shard_key": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "",
                "title": "Shard Key",
            },
        },
        "required": ["filter"],
        "title": "FilterSelector",
        "type": "object",
    },
    "GeoBoundingBox": {
        "additionalProperties": False,
        "description": "Geo filter request  Matches coordinates inside the rectangle, described by coordinates of lop-left and bottom-right edges",
        "properties": {
            "top_left": {
                "allOf": [{"$ref": "#/$defs/GeoPoint"}],
                "description": "Geo filter request  Matches coordinates inside the rectangle, described by coordinates of lop-left and bottom-right edges",
            },
            "bottom_right": {
                "allOf": [{"$ref": "#/$defs/GeoPoint"}],
                "description": "Geo filter request  Matches coordinates inside the rectangle, described by coordinates of lop-left and bottom-right edges",
            },
        },
        "required": ["top_left", "bottom_right"],
        "title": "GeoBoundingBox",
        "type": "object",
    },
    "GeoLineString": {
        "additionalProperties": False,
        "description": "Ordered sequence of GeoPoints representing the line",
        "properties": {
            "points": {
                "description": "Ordered sequence of GeoPoints representing the line",
                "items": {"$ref": "#/$defs/GeoPoint"},
                "title": "Points",
                "type": "array",
            }
        },
        "required": ["points"],
        "title": "GeoLineString",
        "type": "object",
    },
    "GeoPoint": {
        "additionalProperties": False,
        "description": "Geo point payload schema",
        "properties": {
            "lon": {"description": "Geo point payload schema", "title": "Lon", "type": "number"},
            "lat": {"description": "Geo point payload schema", "title": "Lat", "type": "number"},
        },
        "required": ["lon", "lat"],
        "title": "GeoPoint",
        "type": "object",
    },
    "GeoPolygon": {
        "additionalProperties": False,
        "description": "Geo filter request  Matches coordinates inside the polygon, defined by `exterior` and `interiors`",
        "properties": {
            "exterior": {
                "allOf": [{"$ref": "#/$defs/GeoLineString"}],
                "description": "Geo filter request  Matches coordinates inside the polygon, defined by `exterior` and `interiors`",
            },
            "interiors": {
                "anyOf": [
                    {"items": {"$ref": "#/$defs/GeoLineString"}, "type": "array"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Interior lines (if present) bound holes within the surface each GeoLineString must consist of a minimum of 4 points, and the first and last points must be the same.",
                "title": "Interiors",
            },
        },
        "required": ["exterior"],
        "title": "GeoPolygon",
        "type": "object",
    },
    "GeoRadius": {
        "additionalProperties": False,
        "description": "Geo filter request  Matches coordinates inside the circle of `radius` and center with coordinates `center`",
        "properties": {
            "center": {
                "allOf": [{"$ref": "#/$defs/GeoPoint"}],
                "description": "Geo filter request  Matches coordinates inside the circle of `radius` and center with coordinates `center`",
            },
            "radius": {
                "description": "Radius of the area in meters",
                "title": "Radius",
                "type": "number",
            },
        },
        "required": ["center", "radius"],
        "title": "GeoRadius",
        "type": "object",
    },
    "HasIdCondition": {
        "additionalProperties": False,
        "description": "ID-based filtering condition",
        "properties": {
            "has_id": {
                "description": "ID-based filtering condition",
                "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                "title": "Has Id",
                "type": "array",
            }
        },
        "required": ["has_id"],
        "title": "HasIdCondition",
        "type": "object",
    },
    "IsEmptyCondition": {
        "additionalProperties": False,
        "description": "Select points with empty payload for a specified field",
        "properties": {
            "is_empty": {
                "allOf": [{"$ref": "#/$defs/PayloadField"}],
                "description": "Select points with empty payload for a specified field",
            }
        },
        "required": ["is_empty"],
        "title": "IsEmptyCondition",
        "type": "object",
    },
    "IsNullCondition": {
        "additionalProperties": False,
        "description": "Select points with null payload for a specified field",
        "properties": {
            "is_null": {
                "allOf": [{"$ref": "#/$defs/PayloadField"}],
                "description": "Select points with null payload for a specified field",
            }
        },
        "required": ["is_null"],
        "title": "IsNullCondition",
        "type": "object",
    },
    "MatchAny": {
        "additionalProperties": False,
        "description": "Exact match on any of the given values",
        "properties": {
            "any": {
                "anyOf": [
                    {"items": {"type": "string"}, "type": "array"},
                    {"items": {"type": "integer"}, "type": "array"},
                ],
                "description": "Exact match on any of the given values",
                "title": "Any",
            }
        },
        "required": ["any"],
        "title": "MatchAny",
        "type": "object",
    },
    "MatchExcept": {
        "additionalProperties": False,
        "description": "Should have at least one value not matching the any given values",
        "properties": {
            "except": {
                "anyOf": [
                    {"items": {"type": "string"}, "type": "array"},
                    {"items": {"type": "integer"}, "type": "array"},
                ],
                "description": "Should have at least one value not matching the any given values",
                "title": "Except",
            }
        },
        "required": ["except"],
        "title": "MatchExcept",
        "type": "object",
    },
    "MatchText": {
        "additionalProperties": False,
        "description": "Full-text match of the strings.",
        "properties": {
            "text": {
                "description": "Full-text match of the strings.",
                "title": "Text",
                "type": "string",
            }
        },
        "required": ["text"],
        "title": "MatchText",
        "type": "object",
    },
    "MatchValue": {
        "additionalProperties": False,
        "description": "Exact match of the given value",
        "properties": {
            "value": {
                "anyOf": [{"type": "boolean"}, {"type": "integer"}, {"type": "string"}],
                "description": "Exact match of the given value",
                "title": "Value",
            }
        },
        "required": ["value"],
        "title": "MatchValue",
        "type": "object",
    },
    "MinShould": {
        "additionalProperties": False,
        "properties": {
            "conditions": {
                "description": "",
                "items": {
                    "anyOf": [
                        {"$ref": "#/$defs/FieldCondition"},
                        {"$ref": "#/$defs/IsEmptyCondition"},
                        {"$ref": "#/$defs/IsNullCondition"},
                        {"$ref": "#/$defs/HasIdCondition"},
                        {"$ref": "#/$defs/NestedCondition"},
                        {"$ref": "#/$defs/Filter"},
                    ]
                },
                "title": "Conditions",
                "type": "array",
            },
            "min_count": {"description": "", "title": "Min Count", "type": "integer"},
        },
        "required": ["conditions", "min_count"],
        "title": "MinShould",
        "type": "object",
    },
    "Nested": {
        "additionalProperties": False,
        "description": "Select points with payload for a specified nested field",
        "properties": {
            "key": {
                "description": "Select points with payload for a specified nested field",
                "title": "Key",
                "type": "string",
            },
            "filter": {
                "allOf": [{"$ref": "#/$defs/Filter"}],
                "description": "Select points with payload for a specified nested field",
            },
        },
        "required": ["key", "filter"],
        "title": "Nested",
        "type": "object",
    },
    "NestedCondition": {
        "additionalProperties": False,
        "properties": {"nested": {"allOf": [{"$ref": "#/$defs/Nested"}], "description": ""}},
        "required": ["nested"],
        "title": "NestedCondition",
        "type": "object",
    },
    "PayloadField": {
        "additionalProperties": False,
        "description": "Payload field",
        "properties": {
            "key": {"description": "Payload field name", "title": "Key", "type": "string"}
        },
        "required": ["key"],
        "title": "PayloadField",
        "type": "object",
    },
    "PointIdsList": {
        "additionalProperties": False,
        "properties": {
            "points": {
                "description": "",
                "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                "title": "Points",
                "type": "array",
            },
            "shard_key": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "",
                "title": "Shard Key",
            },
        },
        "required": ["points"],
        "title": "PointIdsList",
        "type": "object",
    },
    "Range": {
        "additionalProperties": False,
        "description": "Range filter request",
        "properties": {
            "lt": {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "default": None,
                "description": "point.key &lt; range.lt",
                "title": "Lt",
            },
            "gt": {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "default": None,
                "description": "point.key &gt; range.gt",
                "title": "Gt",
            },
            "gte": {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "default": None,
                "description": "point.key &gt;= range.gte",
                "title": "Gte",
            },
            "lte": {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "default": None,
                "description": "point.key &lt;= range.lte",
                "title": "Lte",
            },
        },
        "title": "Range",
        "type": "object",
    },
    "ValuesCount": {
        "additionalProperties": False,
        "description": "Values count filter request",
        "properties": {
            "lt": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "point.key.length() &lt; values_count.lt",
                "title": "Lt",
            },
            "gt": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "point.key.length() &gt; values_count.gt",
                "title": "Gt",
            },
            "gte": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "point.key.length() &gt;= values_count.gte",
                "title": "Gte",
            },
            "lte": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "point.key.length() &lt;= values_count.lte",
                "title": "Lte",
            },
        },
        "title": "ValuesCount",
        "type": "object",
    },
    "ContextPair": {
        "additionalProperties": False,
        "properties": {
            "positive": {
                "anyOf": [
                    {"items": {"type": "number"}, "type": "array"},
                    {"$ref": "#/$defs/SparseVector"},
                    {"items": {"items": {"type": "number"}, "type": "array"}, "type": "array"},
                    {"type": "integer"},
                    {"type": "string"},
                    {"$ref": "#/$defs/Document"},
                ],
                "description": "",
                "title": "Positive",
            },
            "negative": {
                "anyOf": [
                    {"items": {"type": "number"}, "type": "array"},
                    {"$ref": "#/$defs/SparseVector"},
                    {"items": {"items": {"type": "number"}, "type": "array"}, "type": "array"},
                    {"type": "integer"},
                    {"type": "string"},
                    {"$ref": "#/$defs/Document"},
                ],
                "description": "",
                "title": "Negative",
            },
        },
        "required": ["positive", "negative"],
        "title": "ContextPair",
        "type": "object",
    },
    "BinaryQuantization": {
        "additionalProperties": False,
        "properties": {
            "binary": {"allOf": [{"$ref": "#/$defs/BinaryQuantizationConfig"}], "description": ""}
        },
        "required": ["binary"],
        "title": "BinaryQuantization",
        "type": "object",
    },
    "CompressionRatio": {
        "enum": ["x4", "x8", "x16", "x32", "x64"],
        "title": "CompressionRatio",
        "type": "string",
    },
    "Datatype": {"enum": ["float32", "uint8", "float16"], "title": "Datatype", "type": "string"},
    "Distance": {
        "description": "Type of internal tags, build from payload Distance function types used to compare vectors",
        "enum": ["Cosine", "Euclid", "Dot", "Manhattan"],
        "title": "Distance",
        "type": "string",
    },
    "HnswConfigDiff": {
        "additionalProperties": False,
        "properties": {
            "m": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Number of edges per node in the index graph. Larger the value - more accurate the search, more space required.",
                "title": "M",
            },
            "ef_construct": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Number of neighbours to consider during the index building. Larger the value - more accurate the search, more time required to build the index.",
                "title": "Ef Construct",
            },
            "full_scan_threshold": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Minimal size (in kilobytes) of vectors for additional payload-based indexing. If payload chunk is smaller than `full_scan_threshold_kb` additional indexing won&#x27;t be used - in this case full-scan search should be preferred by query planner and additional indexing is not required. Note: 1Kb = 1 vector of size 256",
                "title": "Full Scan Threshold",
            },
            "max_indexing_threads": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Number of parallel threads used for background index building. If 0 - automatically select from 8 to 16. Best to keep between 8 and 16 to prevent likelihood of building broken/inefficient HNSW graphs. On small CPUs, less threads are used.",
                "title": "Max Indexing Threads",
            },
            "on_disk": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "Store HNSW index on disk. If set to false, the index will be stored in RAM. Default: false",
                "title": "On Disk",
            },
            "payload_m": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Custom M param for additional payload-aware HNSW links. If not set, default M will be used.",
                "title": "Payload M",
            },
        },
        "title": "HnswConfigDiff",
        "type": "object",
    },
    "InitFrom": {
        "additionalProperties": False,
        "description": "Operation for creating new collection and (optionally) specify index params",
        "properties": {
            "collection": {
                "description": "Operation for creating new collection and (optionally) specify index params",
                "title": "Collection",
                "type": "string",
            }
        },
        "required": ["collection"],
        "title": "InitFrom",
        "type": "object",
    },
    "Modifier": {
        "description": "If used, include weight modification, which will be applied to sparse vectors at query time: None - no modification (default) Idf - inverse document frequency, based on statistics of the collection",
        "enum": ["none", "idf"],
        "title": "Modifier",
        "type": "string",
    },
    "MultiVectorComparator": {
        "const": "max_sim",
        "enum": ["max_sim"],
        "title": "MultiVectorComparator",
        "type": "string",
    },
    "MultiVectorConfig": {
        "additionalProperties": False,
        "properties": {
            "comparator": {"allOf": [{"$ref": "#/$defs/MultiVectorComparator"}], "description": ""}
        },
        "required": ["comparator"],
        "title": "MultiVectorConfig",
        "type": "object",
    },
    "OptimizersConfigDiff": {
        "additionalProperties": False,
        "properties": {
            "deleted_threshold": {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "default": None,
                "description": "The minimal fraction of deleted vectors in a segment, required to perform segment optimization",
                "title": "Deleted Threshold",
            },
            "vacuum_min_vector_number": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "The minimal number of vectors in a segment, required to perform segment optimization",
                "title": "Vacuum Min Vector Number",
            },
            "default_segment_number": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Target amount of segments optimizer will try to keep. Real amount of segments may vary depending on multiple parameters: - Amount of stored points - Current write RPS  It is recommended to select default number of segments as a factor of the number of search threads, so that each segment would be handled evenly by one of the threads If `default_segment_number = 0`, will be automatically selected by the number of available CPUs",
                "title": "Default Segment Number",
            },
            "max_segment_size": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Do not create segments larger this size (in kilobytes). Large segments might require disproportionately long indexation times, therefore it makes sense to limit the size of segments.  If indexation speed have more priority for your - make this parameter lower. If search speed is more important - make this parameter higher. Note: 1Kb = 1 vector of size 256",
                "title": "Max Segment Size",
            },
            "memmap_threshold": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Maximum size (in kilobytes) of vectors to store in-memory per segment. Segments larger than this threshold will be stored as read-only memmaped file.  Memmap storage is disabled by default, to enable it, set this threshold to a reasonable value.  To disable memmap storage, set this to `0`.  Note: 1Kb = 1 vector of size 256",
                "title": "Memmap Threshold",
            },
            "indexing_threshold": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Maximum size (in kilobytes) of vectors allowed for plain index, exceeding this threshold will enable vector indexing  Default value is 20,000, based on &lt;https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md&gt;.  To disable vector indexing, set to `0`.  Note: 1kB = 1 vector of size 256.",
                "title": "Indexing Threshold",
            },
            "flush_interval_sec": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Minimum interval between forced flushes.",
                "title": "Flush Interval Sec",
            },
            "max_optimization_threads": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Max number of threads (jobs) for running optimizations per shard. Note: each optimization job will also use `max_indexing_threads` threads by itself for index building. If null - have no limit and choose dynamically to saturate CPU. If 0 - no optimization threads, optimizations will be disabled.",
                "title": "Max Optimization Threads",
            },
        },
        "title": "OptimizersConfigDiff",
        "type": "object",
    },
    "ProductQuantization": {
        "additionalProperties": False,
        "properties": {
            "product": {
                "allOf": [{"$ref": "#/$defs/ProductQuantizationConfig"}],
                "description": "",
            }
        },
        "required": ["product"],
        "title": "ProductQuantization",
        "type": "object",
    },
    "ProductQuantizationConfig": {
        "additionalProperties": False,
        "properties": {
            "compression": {"allOf": [{"$ref": "#/$defs/CompressionRatio"}], "description": ""},
            "always_ram": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "",
                "title": "Always Ram",
            },
        },
        "required": ["compression"],
        "title": "ProductQuantizationConfig",
        "type": "object",
    },
    "ScalarQuantization": {
        "additionalProperties": False,
        "properties": {
            "scalar": {"allOf": [{"$ref": "#/$defs/ScalarQuantizationConfig"}], "description": ""}
        },
        "required": ["scalar"],
        "title": "ScalarQuantization",
        "type": "object",
    },
    "ScalarQuantizationConfig": {
        "additionalProperties": False,
        "properties": {
            "type": {"allOf": [{"$ref": "#/$defs/ScalarType"}], "description": ""},
            "quantile": {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "default": None,
                "description": "Quantile for quantization. Expected value range in [0.5, 1.0]. If not set - use the whole range of values",
                "title": "Quantile",
            },
            "always_ram": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true - quantized vectors always will be stored in RAM, ignoring the config of main storage",
                "title": "Always Ram",
            },
        },
        "required": ["type"],
        "title": "ScalarQuantizationConfig",
        "type": "object",
    },
    "ScalarType": {"const": "int8", "enum": ["int8"], "title": "ScalarType", "type": "string"},
    "ShardingMethod": {"enum": ["auto", "custom"], "title": "ShardingMethod", "type": "string"},
    "SparseIndexParams": {
        "additionalProperties": False,
        "description": "Configuration for sparse inverted index.",
        "properties": {
            "full_scan_threshold": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "We prefer a full scan search upto (excluding) this number of vectors.  Note: this is number of vectors, not KiloBytes.",
                "title": "Full Scan Threshold",
            },
            "on_disk": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "Store index on disk. If set to false, the index will be stored in RAM. Default: false",
                "title": "On Disk",
            },
            "datatype": {
                "anyOf": [{"$ref": "#/$defs/Datatype"}, {"type": "null"}],
                "default": None,
                "description": "Defines which datatype should be used for the index. Choosing different datatypes allows to optimize memory usage and performance vs accuracy.  - For `float32` datatype - vectors are stored as single-precision floating point numbers, 4 bytes. - For `float16` datatype - vectors are stored as half-precision floating point numbers, 2 bytes. - For `uint8` datatype - vectors are quantized to unsigned 8-bit integers, 1 byte. Quantization to fit byte range `[0, 255]` happens during indexing automatically, so the actual vector data does not need to conform to this range.",
            },
        },
        "title": "SparseIndexParams",
        "type": "object",
    },
    "SparseVectorParams": {
        "additionalProperties": False,
        "description": "Params of single sparse vector data storage",
        "properties": {
            "index": {
                "anyOf": [{"$ref": "#/$defs/SparseIndexParams"}, {"type": "null"}],
                "default": None,
                "description": "Custom params for index. If none - values from collection configuration are used.",
            },
            "modifier": {
                "anyOf": [{"$ref": "#/$defs/Modifier"}, {"type": "null"}],
                "default": None,
                "description": "Configures addition value modifications for sparse vectors. Default: none",
            },
        },
        "title": "SparseVectorParams",
        "type": "object",
    },
    "VectorParams": {
        "additionalProperties": False,
        "description": "Params of single vector data storage",
        "properties": {
            "size": {"description": "Size of a vectors used", "title": "Size", "type": "integer"},
            "distance": {
                "allOf": [{"$ref": "#/$defs/Distance"}],
                "description": "Params of single vector data storage",
            },
            "hnsw_config": {
                "anyOf": [{"$ref": "#/$defs/HnswConfigDiff"}, {"type": "null"}],
                "default": None,
                "description": "Custom params for HNSW index. If none - values from collection configuration are used.",
            },
            "quantization_config": {
                "anyOf": [
                    {"$ref": "#/$defs/ScalarQuantization"},
                    {"$ref": "#/$defs/ProductQuantization"},
                    {"$ref": "#/$defs/BinaryQuantization"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Custom params for quantization. If none - values from collection configuration are used.",
                "title": "Quantization Config",
            },
            "on_disk": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true, vectors are served from disk, improving RAM usage at the cost of latency Default: false",
                "title": "On Disk",
            },
            "datatype": {
                "anyOf": [{"$ref": "#/$defs/Datatype"}, {"type": "null"}],
                "default": None,
                "description": "Defines which datatype should be used to represent vectors in the storage. Choosing different datatypes allows to optimize memory usage and performance vs accuracy.  - For `float32` datatype - vectors are stored as single-precision floating point numbers, 4 bytes. - For `float16` datatype - vectors are stored as half-precision floating point numbers, 2 bytes. - For `uint8` datatype - vectors are stored as unsigned 8-bit integers, 1 byte. It expects vector elements to be in range `[0, 255]`.",
            },
            "multivector_config": {
                "anyOf": [{"$ref": "#/$defs/MultiVectorConfig"}, {"type": "null"}],
                "default": None,
                "description": "Params of single vector data storage",
            },
        },
        "required": ["size", "distance"],
        "title": "VectorParams",
        "type": "object",
    },
    "WalConfigDiff": {
        "additionalProperties": False,
        "properties": {
            "wal_capacity_mb": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Size of a single WAL segment in MB",
                "title": "Wal Capacity Mb",
            },
            "wal_segments_ahead": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Number of WAL segments to create ahead of actually used ones",
                "title": "Wal Segments Ahead",
            },
        },
        "title": "WalConfigDiff",
        "type": "object",
    },
    "BoolIndexParams": {
        "additionalProperties": False,
        "properties": {"type": {"allOf": [{"$ref": "#/$defs/BoolIndexType"}], "description": ""}},
        "required": ["type"],
        "title": "BoolIndexParams",
        "type": "object",
    },
    "DatetimeIndexParams": {
        "additionalProperties": False,
        "properties": {
            "type": {"allOf": [{"$ref": "#/$defs/DatetimeIndexType"}], "description": ""},
            "is_principal": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true - use this key to organize storage of the collection data. This option assumes that this key will be used in majority of filtered requests.",
                "title": "Is Principal",
            },
            "on_disk": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true, store the index on disk. Default: false.",
                "title": "On Disk",
            },
        },
        "required": ["type"],
        "title": "DatetimeIndexParams",
        "type": "object",
    },
    "DatetimeIndexType": {
        "const": "datetime",
        "enum": ["datetime"],
        "title": "DatetimeIndexType",
        "type": "string",
    },
    "FloatIndexParams": {
        "additionalProperties": False,
        "properties": {
            "type": {"allOf": [{"$ref": "#/$defs/FloatIndexType"}], "description": ""},
            "is_principal": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true - use this key to organize storage of the collection data. This option assumes that this key will be used in majority of filtered requests.",
                "title": "Is Principal",
            },
            "on_disk": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true, store the index on disk. Default: false.",
                "title": "On Disk",
            },
        },
        "required": ["type"],
        "title": "FloatIndexParams",
        "type": "object",
    },
    "FloatIndexType": {
        "const": "float",
        "enum": ["float"],
        "title": "FloatIndexType",
        "type": "string",
    },
    "GeoIndexParams": {
        "additionalProperties": False,
        "properties": {
            "type": {"allOf": [{"$ref": "#/$defs/GeoIndexType"}], "description": ""},
            "on_disk": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true, store the index on disk. Default: false.",
                "title": "On Disk",
            },
        },
        "required": ["type"],
        "title": "GeoIndexParams",
        "type": "object",
    },
    "GeoIndexType": {"const": "geo", "enum": ["geo"], "title": "GeoIndexType", "type": "string"},
    "IntegerIndexParams": {
        "additionalProperties": False,
        "properties": {
            "type": {"allOf": [{"$ref": "#/$defs/IntegerIndexType"}], "description": ""},
            "lookup": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true - support direct lookups.",
                "title": "Lookup",
            },
            "range": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true - support ranges filters.",
                "title": "Range",
            },
            "is_principal": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true - use this key to organize storage of the collection data. This option assumes that this key will be used in majority of filtered requests.",
                "title": "Is Principal",
            },
            "on_disk": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true, store the index on disk. Default: false.",
                "title": "On Disk",
            },
        },
        "required": ["type"],
        "title": "IntegerIndexParams",
        "type": "object",
    },
    "IntegerIndexType": {
        "const": "integer",
        "enum": ["integer"],
        "title": "IntegerIndexType",
        "type": "string",
    },
    "KeywordIndexParams": {
        "additionalProperties": False,
        "properties": {
            "type": {"allOf": [{"$ref": "#/$defs/KeywordIndexType"}], "description": ""},
            "is_tenant": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true - used for tenant optimization. Default: false.",
                "title": "Is Tenant",
            },
            "on_disk": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true, store the index on disk. Default: false.",
                "title": "On Disk",
            },
        },
        "required": ["type"],
        "title": "KeywordIndexParams",
        "type": "object",
    },
    "KeywordIndexType": {
        "const": "keyword",
        "enum": ["keyword"],
        "title": "KeywordIndexType",
        "type": "string",
    },
    "PayloadSchemaType": {
        "description": "All possible names of payload types",
        "enum": ["keyword", "integer", "float", "geo", "text", "bool", "datetime", "uuid"],
        "title": "PayloadSchemaType",
        "type": "string",
    },
    "TextIndexParams": {
        "additionalProperties": False,
        "properties": {
            "type": {"allOf": [{"$ref": "#/$defs/TextIndexType"}], "description": ""},
            "tokenizer": {
                "anyOf": [{"$ref": "#/$defs/TokenizerType"}, {"type": "null"}],
                "default": None,
                "description": "",
            },
            "min_token_len": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Minimum characters to be tokenized.",
                "title": "Min Token Len",
            },
            "max_token_len": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Maximum characters to be tokenized.",
                "title": "Max Token Len",
            },
            "lowercase": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true, lowercase all tokens. Default: true.",
                "title": "Lowercase",
            },
            "on_disk": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true, store the index on disk. Default: false.",
                "title": "On Disk",
            },
        },
        "required": ["type"],
        "title": "TextIndexParams",
        "type": "object",
    },
    "TextIndexType": {
        "const": "text",
        "enum": ["text"],
        "title": "TextIndexType",
        "type": "string",
    },
    "TokenizerType": {
        "enum": ["prefix", "whitespace", "word", "multilingual"],
        "title": "TokenizerType",
        "type": "string",
    },
    "UuidIndexParams": {
        "additionalProperties": False,
        "properties": {
            "type": {"allOf": [{"$ref": "#/$defs/UuidIndexType"}], "description": ""},
            "is_tenant": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true - used for tenant optimization.",
                "title": "Is Tenant",
            },
            "on_disk": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true, store the index on disk. Default: false.",
                "title": "On Disk",
            },
        },
        "required": ["type"],
        "title": "UuidIndexParams",
        "type": "object",
    },
    "UuidIndexType": {
        "const": "uuid",
        "enum": ["uuid"],
        "title": "UuidIndexType",
        "type": "string",
    },
    "CreateShardingKey": {
        "additionalProperties": False,
        "properties": {
            "shard_key": {
                "anyOf": [{"type": "integer"}, {"type": "string"}],
                "description": "",
                "title": "Shard Key",
            },
            "shards_number": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "How many shards to create for this key If not specified, will use the default value from config",
                "title": "Shards Number",
            },
            "replication_factor": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "How many replicas to create for each shard If not specified, will use the default value from config",
                "title": "Replication Factor",
            },
            "placement": {
                "anyOf": [{"items": {"type": "integer"}, "type": "array"}, {"type": "null"}],
                "default": None,
                "description": "Placement of shards for this key List of peer ids, that can be used to place shards for this key If not specified, will be randomly placed among all peers",
                "title": "Placement",
            },
        },
        "required": ["shard_key"],
        "title": "CreateShardingKey",
        "type": "object",
    },
    "DeletePayload": {
        "additionalProperties": False,
        "description": "This data structure is used in API interface and applied across multiple shards",
        "properties": {
            "keys": {
                "description": "List of payload keys to remove from payload",
                "items": {"type": "string"},
                "title": "Keys",
                "type": "array",
            },
            "points": {
                "anyOf": [
                    {
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "Deletes values from each point in this list",
                "title": "Points",
            },
            "filter": {
                "anyOf": [{"$ref": "#/$defs/Filter"}, {"type": "null"}],
                "default": None,
                "description": "Deletes values from points that satisfy this filter condition",
            },
            "shard_key": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "This data structure is used in API interface and applied across multiple shards",
                "title": "Shard Key",
            },
        },
        "required": ["keys"],
        "title": "DeletePayload",
        "type": "object",
    },
    "DeleteVectors": {
        "additionalProperties": False,
        "properties": {
            "points": {
                "anyOf": [
                    {
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "Deletes values from each point in this list",
                "title": "Points",
            },
            "filter": {
                "anyOf": [{"$ref": "#/$defs/Filter"}, {"type": "null"}],
                "default": None,
                "description": "Deletes values from points that satisfy this filter condition",
            },
            "vector": {
                "description": "Vector names",
                "items": {"type": "string"},
                "title": "Vector",
                "type": "array",
            },
            "shard_key": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "",
                "title": "Shard Key",
            },
        },
        "required": ["vector"],
        "title": "DeleteVectors",
        "type": "object",
    },
    "DiscoverInput": {
        "additionalProperties": False,
        "properties": {
            "target": {
                "anyOf": [
                    {"items": {"type": "number"}, "type": "array"},
                    {"$ref": "#/$defs/SparseVector"},
                    {"items": {"items": {"type": "number"}, "type": "array"}, "type": "array"},
                    {"type": "integer"},
                    {"type": "string"},
                    {"$ref": "#/$defs/Document"},
                ],
                "description": "",
                "title": "Target",
            },
            "context": {
                "anyOf": [
                    {"items": {"$ref": "#/$defs/ContextPair"}, "type": "array"},
                    {"$ref": "#/$defs/ContextPair"},
                ],
                "description": "Search space will be constrained by these pairs of vectors",
                "title": "Context",
            },
        },
        "required": ["target", "context"],
        "title": "DiscoverInput",
        "type": "object",
    },
    "ContextExamplePair": {
        "additionalProperties": False,
        "properties": {
            "positive": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {"items": {"type": "number"}, "type": "array"},
                    {"$ref": "#/$defs/SparseVector"},
                ],
                "description": "",
                "title": "Positive",
            },
            "negative": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {"items": {"type": "number"}, "type": "array"},
                    {"$ref": "#/$defs/SparseVector"},
                ],
                "description": "",
                "title": "Negative",
            },
        },
        "required": ["positive", "negative"],
        "title": "ContextExamplePair",
        "type": "object",
    },
    "LookupLocation": {
        "additionalProperties": False,
        "description": "Defines a location to use for looking up the vector. Specifies collection and vector field name.",
        "properties": {
            "collection": {
                "description": "Name of the collection used for lookup",
                "title": "Collection",
                "type": "string",
            },
            "vector": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "description": "Optional name of the vector field within the collection. If not provided, the default vector field will be used.",
                "title": "Vector",
            },
            "shard_key": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "Specify in which shards to look for the points, if not specified - look in all shards",
                "title": "Shard Key",
            },
        },
        "required": ["collection"],
        "title": "LookupLocation",
        "type": "object",
    },
    "PayloadSelectorExclude": {
        "additionalProperties": False,
        "properties": {
            "exclude": {
                "description": "Exclude this fields from returning payload",
                "items": {"type": "string"},
                "title": "Exclude",
                "type": "array",
            }
        },
        "required": ["exclude"],
        "title": "PayloadSelectorExclude",
        "type": "object",
    },
    "PayloadSelectorInclude": {
        "additionalProperties": False,
        "properties": {
            "include": {
                "description": "Only include this payload keys",
                "items": {"type": "string"},
                "title": "Include",
                "type": "array",
            }
        },
        "required": ["include"],
        "title": "PayloadSelectorInclude",
        "type": "object",
    },
    "QuantizationSearchParams": {
        "additionalProperties": False,
        "description": "Additional parameters of the search",
        "properties": {
            "ignore": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": False,
                "description": "If true, quantized vectors are ignored. Default is false.",
                "title": "Ignore",
            },
            "rescore": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true, use original vectors to re-score top-k results. Might require more time in case if original vectors are stored on disk. If not set, qdrant decides automatically apply rescoring or not.",
                "title": "Rescore",
            },
            "oversampling": {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "default": None,
                "description": "Oversampling factor for quantization. Default is 1.0.  Defines how many extra vectors should be pre-selected using quantized index, and then re-scored using original vectors.  For example, if `oversampling` is 2.4 and `limit` is 100, then 240 vectors will be pre-selected using quantized index, and then top-100 will be returned after re-scoring.",
                "title": "Oversampling",
            },
        },
        "title": "QuantizationSearchParams",
        "type": "object",
    },
    "SearchParams": {
        "additionalProperties": False,
        "description": "Additional parameters of the search",
        "properties": {
            "hnsw_ef": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Params relevant to HNSW index Size of the beam in a beam-search. Larger the value - more accurate the result, more time required for search.",
                "title": "Hnsw Ef",
            },
            "exact": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": False,
                "description": "Search without approximation. If set to true, search may run long but with exact results.",
                "title": "Exact",
            },
            "quantization": {
                "anyOf": [{"$ref": "#/$defs/QuantizationSearchParams"}, {"type": "null"}],
                "default": None,
                "description": "Quantization params",
            },
            "indexed_only": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": False,
                "description": "If enabled, the engine will only perform search among indexed or small segments. Using this option prevents slow searches in case of delayed index, but does not guarantee that all uploaded vectors will be included in search results",
                "title": "Indexed Only",
            },
        },
        "title": "SearchParams",
        "type": "object",
    },
    "DiscoverRequest": {
        "additionalProperties": False,
        "description": "Use context and a target to find the most similar points, constrained by the context.",
        "properties": {
            "shard_key": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "Specify in which shards to look for the points, if not specified - look in all shards",
                "title": "Shard Key",
            },
            "target": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {"items": {"type": "number"}, "type": "array"},
                    {"$ref": "#/$defs/SparseVector"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Look for vectors closest to this.  When using the target (with or without context), the integer part of the score represents the rank with respect to the context, while the decimal part of the score relates to the distance to the target.",
                "title": "Target",
            },
            "context": {
                "anyOf": [
                    {"items": {"$ref": "#/$defs/ContextExamplePair"}, "type": "array"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Pairs of { positive, negative } examples to constrain the search.  When using only the context (without a target), a special search - called context search - is performed where pairs of points are used to generate a loss that guides the search towards the zone where most positive examples overlap. This means that the score minimizes the scenario of finding a point closer to a negative than to a positive part of a pair.  Since the score of a context relates to loss, the maximum score a point can get is 0.0, and it becomes normal that many points can have a score of 0.0.  For discovery search (when including a target), the context part of the score for each pair is calculated +1 if the point is closer to a positive than to a negative part of a pair, and -1 otherwise.",
                "title": "Context",
            },
            "filter": {
                "anyOf": [{"$ref": "#/$defs/Filter"}, {"type": "null"}],
                "default": None,
                "description": "Look only for points which satisfies this conditions",
            },
            "params": {
                "anyOf": [{"$ref": "#/$defs/SearchParams"}, {"type": "null"}],
                "default": None,
                "description": "Additional search params",
            },
            "limit": {
                "description": "Max number of result to return",
                "title": "Limit",
                "type": "integer",
            },
            "offset": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Offset of the first result to return. May be used to paginate results. Note: large offset values may cause performance issues.",
                "title": "Offset",
            },
            "with_payload": {
                "anyOf": [
                    {"type": "boolean"},
                    {"items": {"type": "string"}, "type": "array"},
                    {"$ref": "#/$defs/PayloadSelectorInclude"},
                    {"$ref": "#/$defs/PayloadSelectorExclude"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Select which payload to return with the response. Default is false.",
                "title": "With Payload",
            },
            "with_vector": {
                "anyOf": [
                    {"type": "boolean"},
                    {"items": {"type": "string"}, "type": "array"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Options for specifying which vectors to include into response. Default is false.",
                "title": "With Vector",
            },
            "using": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "description": "Define which vector to use for recommendation, if not specified - try to use default vector",
                "title": "Using",
            },
            "lookup_from": {
                "anyOf": [{"$ref": "#/$defs/LookupLocation"}, {"type": "null"}],
                "default": None,
                "description": "The location used to lookup vectors. If not specified - use current collection. Note: the other collection should have the same vector size as the current collection",
            },
        },
        "required": ["limit"],
        "title": "DiscoverRequest",
        "type": "object",
    },
    "Replica": {
        "additionalProperties": False,
        "properties": {
            "shard_id": {"description": "", "title": "Shard Id", "type": "integer"},
            "peer_id": {"description": "", "title": "Peer Id", "type": "integer"},
        },
        "required": ["shard_id", "peer_id"],
        "title": "Replica",
        "type": "object",
    },
    "DropShardingKey": {
        "additionalProperties": False,
        "properties": {
            "shard_key": {
                "anyOf": [{"type": "integer"}, {"type": "string"}],
                "description": "",
                "title": "Shard Key",
            }
        },
        "required": ["shard_key"],
        "title": "DropShardingKey",
        "type": "object",
    },
    "Fusion": {
        "description": "Fusion algorithm allows to combine results of multiple prefetches.  Available fusion algorithms:  * `rrf` - Reciprocal Rank Fusion * `dbsf` - Distribution-Based Score Fusion",
        "enum": ["rrf", "dbsf"],
        "title": "Fusion",
        "type": "string",
    },
    "ShardTransferMethodOneOf": {
        "const": "stream_records",
        "description": "Stream all shard records in batches until the whole shard is transferred.",
        "enum": ["stream_records"],
        "title": "ShardTransferMethodOneOf",
        "type": "string",
    },
    "ShardTransferMethodOneOf1": {
        "const": "snapshot",
        "description": "Snapshot the shard, transfer and restore it on the receiver.",
        "enum": ["snapshot"],
        "title": "ShardTransferMethodOneOf1",
        "type": "string",
    },
    "ShardTransferMethodOneOf2": {
        "const": "wal_delta",
        "description": "Attempt to transfer shard difference by WAL delta.",
        "enum": ["wal_delta"],
        "title": "ShardTransferMethodOneOf2",
        "type": "string",
    },
    "MoveShard": {
        "additionalProperties": False,
        "properties": {
            "shard_id": {"description": "", "title": "Shard Id", "type": "integer"},
            "to_peer_id": {"description": "", "title": "To Peer Id", "type": "integer"},
            "from_peer_id": {"description": "", "title": "From Peer Id", "type": "integer"},
            "method": {
                "anyOf": [
                    {"$ref": "#/$defs/ShardTransferMethodOneOf"},
                    {"$ref": "#/$defs/ShardTransferMethodOneOf1"},
                    {"$ref": "#/$defs/ShardTransferMethodOneOf2"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Method for transferring the shard from one node to another",
                "title": "Method",
            },
        },
        "required": ["shard_id", "to_peer_id", "from_peer_id"],
        "title": "MoveShard",
        "type": "object",
    },
    "Direction": {"enum": ["asc", "desc"], "title": "Direction", "type": "string"},
    "OrderBy": {
        "additionalProperties": False,
        "properties": {
            "key": {"description": "Payload key to order by", "title": "Key", "type": "string"},
            "direction": {
                "anyOf": [{"$ref": "#/$defs/Direction"}, {"type": "null"}],
                "default": None,
                "description": "Direction of ordering: `asc` or `desc`. Default is ascending.",
            },
            "start_from": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "number"},
                    {"format": "date-time", "type": "string"},
                    {"format": "date", "type": "string"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Which payload value to start scrolling from. Default is the lowest value for `asc` and the highest for `desc`",
                "title": "Start From",
            },
        },
        "required": ["key"],
        "title": "OrderBy",
        "type": "object",
    },
    "SetPayload": {
        "additionalProperties": False,
        "description": "This data structure is used in API interface and applied across multiple shards",
        "properties": {
            "payload": {
                "description": "This data structure is used in API interface and applied across multiple shards",
                "title": "Payload",
                "type": "object",
            },
            "points": {
                "anyOf": [
                    {
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "Assigns payload to each point in this list",
                "title": "Points",
            },
            "filter": {
                "anyOf": [{"$ref": "#/$defs/Filter"}, {"type": "null"}],
                "default": None,
                "description": "Assigns payload to each point that satisfy this filter condition",
            },
            "shard_key": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "This data structure is used in API interface and applied across multiple shards",
                "title": "Shard Key",
            },
            "key": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "description": "Assigns payload to each point that satisfy this path of property",
                "title": "Key",
            },
        },
        "required": ["payload"],
        "title": "SetPayload",
        "type": "object",
    },
    "Batch": {
        "additionalProperties": False,
        "properties": {
            "ids": {
                "description": "",
                "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                "title": "Ids",
                "type": "array",
            },
            "vectors": {
                "anyOf": [
                    {"items": {"items": {"type": "number"}, "type": "array"}, "type": "array"},
                    {
                        "items": {
                            "items": {"items": {"type": "number"}, "type": "array"},
                            "type": "array",
                        },
                        "type": "array",
                    },
                    {
                        "additionalProperties": {
                            "items": {
                                "anyOf": [
                                    {"items": {"type": "number"}, "type": "array"},
                                    {"$ref": "#/$defs/SparseVector"},
                                    {
                                        "items": {"items": {"type": "number"}, "type": "array"},
                                        "type": "array",
                                    },
                                    {"$ref": "#/$defs/Document"},
                                ]
                            },
                            "type": "array",
                        },
                        "type": "object",
                    },
                    {"items": {"$ref": "#/$defs/Document"}, "type": "array"},
                ],
                "description": "",
                "title": "Vectors",
            },
            "payloads": {
                "anyOf": [{"items": {"type": "object"}, "type": "array"}, {"type": "null"}],
                "default": None,
                "description": "",
                "title": "Payloads",
            },
        },
        "required": ["ids", "vectors"],
        "title": "Batch",
        "type": "object",
    },
    "PointStruct": {
        "additionalProperties": False,
        "properties": {
            "id": {
                "anyOf": [{"type": "integer"}, {"type": "string"}],
                "description": "",
                "title": "Id",
            },
            "vector": {
                "anyOf": [
                    {"items": {"type": "number"}, "type": "array"},
                    {"items": {"items": {"type": "number"}, "type": "array"}, "type": "array"},
                    {
                        "additionalProperties": {
                            "anyOf": [
                                {"items": {"type": "number"}, "type": "array"},
                                {"$ref": "#/$defs/SparseVector"},
                                {
                                    "items": {"items": {"type": "number"}, "type": "array"},
                                    "type": "array",
                                },
                                {"$ref": "#/$defs/Document"},
                            ]
                        },
                        "type": "object",
                    },
                    {"$ref": "#/$defs/Document"},
                ],
                "description": "",
                "title": "Vector",
            },
            "payload": {
                "anyOf": [{"type": "object"}, {"type": "null"}],
                "default": None,
                "description": "Payload values (optional)",
                "title": "Payload",
            },
        },
        "required": ["id", "vector"],
        "title": "PointStruct",
        "type": "object",
    },
    "ContextQuery": {
        "additionalProperties": False,
        "properties": {
            "context": {
                "anyOf": [
                    {"$ref": "#/$defs/ContextPair"},
                    {"items": {"$ref": "#/$defs/ContextPair"}, "type": "array"},
                ],
                "description": "",
                "title": "Context",
            }
        },
        "required": ["context"],
        "title": "ContextQuery",
        "type": "object",
    },
    "DiscoverQuery": {
        "additionalProperties": False,
        "properties": {
            "discover": {"allOf": [{"$ref": "#/$defs/DiscoverInput"}], "description": ""}
        },
        "required": ["discover"],
        "title": "DiscoverQuery",
        "type": "object",
    },
    "FusionQuery": {
        "additionalProperties": False,
        "properties": {"fusion": {"allOf": [{"$ref": "#/$defs/Fusion"}], "description": ""}},
        "required": ["fusion"],
        "title": "FusionQuery",
        "type": "object",
    },
    "NearestQuery": {
        "additionalProperties": False,
        "properties": {
            "nearest": {
                "anyOf": [
                    {"items": {"type": "number"}, "type": "array"},
                    {"$ref": "#/$defs/SparseVector"},
                    {"items": {"items": {"type": "number"}, "type": "array"}, "type": "array"},
                    {"type": "integer"},
                    {"type": "string"},
                    {"$ref": "#/$defs/Document"},
                ],
                "description": "",
                "title": "Nearest",
            }
        },
        "required": ["nearest"],
        "title": "NearestQuery",
        "type": "object",
    },
    "OrderByQuery": {
        "additionalProperties": False,
        "properties": {
            "order_by": {
                "anyOf": [{"type": "string"}, {"$ref": "#/$defs/OrderBy"}],
                "description": "",
                "title": "Order By",
            }
        },
        "required": ["order_by"],
        "title": "OrderByQuery",
        "type": "object",
    },
    "Prefetch": {
        "additionalProperties": False,
        "properties": {
            "prefetch": {
                "anyOf": [
                    {"items": {"$ref": "#/$defs/Prefetch"}, "type": "array"},
                    {"$ref": "#/$defs/Prefetch"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Sub-requests to perform first. If present, the query will be performed on the results of the prefetches.",
                "title": "Prefetch",
            },
            "query": {
                "anyOf": [
                    {"items": {"type": "number"}, "type": "array"},
                    {"$ref": "#/$defs/SparseVector"},
                    {"items": {"items": {"type": "number"}, "type": "array"}, "type": "array"},
                    {"type": "integer"},
                    {"type": "string"},
                    {"$ref": "#/$defs/Document"},
                    {"$ref": "#/$defs/NearestQuery"},
                    {"$ref": "#/$defs/RecommendQuery"},
                    {"$ref": "#/$defs/DiscoverQuery"},
                    {"$ref": "#/$defs/ContextQuery"},
                    {"$ref": "#/$defs/OrderByQuery"},
                    {"$ref": "#/$defs/FusionQuery"},
                    {"$ref": "#/$defs/SampleQuery"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Query to perform. If missing without prefetches, returns points ordered by their IDs.",
                "title": "Query",
            },
            "using": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "description": "Define which vector name to use for querying. If missing, the default vector is used.",
                "title": "Using",
            },
            "filter": {
                "anyOf": [{"$ref": "#/$defs/Filter"}, {"type": "null"}],
                "default": None,
                "description": "Filter conditions - return only those points that satisfy the specified conditions.",
            },
            "params": {
                "anyOf": [{"$ref": "#/$defs/SearchParams"}, {"type": "null"}],
                "default": None,
                "description": "Search params for when there is no prefetch",
            },
            "score_threshold": {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "default": None,
                "description": "Return points with scores better than this threshold.",
                "title": "Score Threshold",
            },
            "limit": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Max number of points to return. Default is 10.",
                "title": "Limit",
            },
            "lookup_from": {
                "anyOf": [{"$ref": "#/$defs/LookupLocation"}, {"type": "null"}],
                "default": None,
                "description": "The location to use for IDs lookup, if not specified - use the current collection and the &#x27;using&#x27; vector Note: the other collection vectors should have the same vector size as the &#x27;using&#x27; vector in the current collection",
            },
        },
        "title": "Prefetch",
        "type": "object",
    },
    "RecommendInput": {
        "additionalProperties": False,
        "properties": {
            "positive": {
                "anyOf": [
                    {
                        "items": {
                            "anyOf": [
                                {"items": {"type": "number"}, "type": "array"},
                                {"$ref": "#/$defs/SparseVector"},
                                {
                                    "items": {"items": {"type": "number"}, "type": "array"},
                                    "type": "array",
                                },
                                {"type": "integer"},
                                {"type": "string"},
                                {"$ref": "#/$defs/Document"},
                            ]
                        },
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "Look for vectors closest to the vectors from these points",
                "title": "Positive",
            },
            "negative": {
                "anyOf": [
                    {
                        "items": {
                            "anyOf": [
                                {"items": {"type": "number"}, "type": "array"},
                                {"$ref": "#/$defs/SparseVector"},
                                {
                                    "items": {"items": {"type": "number"}, "type": "array"},
                                    "type": "array",
                                },
                                {"type": "integer"},
                                {"type": "string"},
                                {"$ref": "#/$defs/Document"},
                            ]
                        },
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "Try to avoid vectors like the vector from these points",
                "title": "Negative",
            },
            "strategy": {
                "anyOf": [{"$ref": "#/$defs/RecommendStrategy"}, {"type": "null"}],
                "default": None,
                "description": "How to use the provided vectors to find the results",
            },
        },
        "title": "RecommendInput",
        "type": "object",
    },
    "RecommendQuery": {
        "additionalProperties": False,
        "properties": {
            "recommend": {"allOf": [{"$ref": "#/$defs/RecommendInput"}], "description": ""}
        },
        "required": ["recommend"],
        "title": "RecommendQuery",
        "type": "object",
    },
    "RecommendStrategy": {
        "description": "How to use positive and negative examples to find the results, default is `average_vector`:  * `average_vector` - Average positive and negative vectors and create a single query with the formula `query = avg_pos + avg_pos - avg_neg`. Then performs normal search.  * `best_score` - Uses custom search objective. Each candidate is compared against all examples, its score is then chosen from the `max(max_pos_score, max_neg_score)`. If the `max_neg_score` is chosen then it is squared and negated, otherwise it is just the `max_pos_score`.",
        "enum": ["average_vector", "best_score"],
        "title": "RecommendStrategy",
        "type": "string",
    },
    "Sample": {"const": "random", "enum": ["random"], "title": "Sample", "type": "string"},
    "SampleQuery": {
        "additionalProperties": False,
        "properties": {"sample": {"allOf": [{"$ref": "#/$defs/Sample"}], "description": ""}},
        "required": ["sample"],
        "title": "SampleQuery",
        "type": "object",
    },
    "WithLookup": {
        "additionalProperties": False,
        "properties": {
            "collection": {
                "description": "Name of the collection to use for points lookup",
                "title": "Collection",
                "type": "string",
            },
            "with_payload": {
                "anyOf": [
                    {"type": "boolean"},
                    {"items": {"type": "string"}, "type": "array"},
                    {"$ref": "#/$defs/PayloadSelectorInclude"},
                    {"$ref": "#/$defs/PayloadSelectorExclude"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Options for specifying which payload to include (or not)",
                "title": "With Payload",
            },
            "with_vectors": {
                "anyOf": [
                    {"type": "boolean"},
                    {"items": {"type": "string"}, "type": "array"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Options for specifying which vectors to include (or not)",
                "title": "With Vectors",
            },
        },
        "required": ["collection"],
        "title": "WithLookup",
        "type": "object",
    },
    "QueryRequest": {
        "additionalProperties": False,
        "properties": {
            "shard_key": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "",
                "title": "Shard Key",
            },
            "prefetch": {
                "anyOf": [
                    {"items": {"$ref": "#/$defs/Prefetch"}, "type": "array"},
                    {"$ref": "#/$defs/Prefetch"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Sub-requests to perform first. If present, the query will be performed on the results of the prefetch(es).",
                "title": "Prefetch",
            },
            "query": {
                "anyOf": [
                    {"items": {"type": "number"}, "type": "array"},
                    {"$ref": "#/$defs/SparseVector"},
                    {"items": {"items": {"type": "number"}, "type": "array"}, "type": "array"},
                    {"type": "integer"},
                    {"type": "string"},
                    {"$ref": "#/$defs/Document"},
                    {"$ref": "#/$defs/NearestQuery"},
                    {"$ref": "#/$defs/RecommendQuery"},
                    {"$ref": "#/$defs/DiscoverQuery"},
                    {"$ref": "#/$defs/ContextQuery"},
                    {"$ref": "#/$defs/OrderByQuery"},
                    {"$ref": "#/$defs/FusionQuery"},
                    {"$ref": "#/$defs/SampleQuery"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Query to perform. If missing without prefetches, returns points ordered by their IDs.",
                "title": "Query",
            },
            "using": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "description": "Define which vector name to use for querying. If missing, the default vector is used.",
                "title": "Using",
            },
            "filter": {
                "anyOf": [{"$ref": "#/$defs/Filter"}, {"type": "null"}],
                "default": None,
                "description": "Filter conditions - return only those points that satisfy the specified conditions.",
            },
            "params": {
                "anyOf": [{"$ref": "#/$defs/SearchParams"}, {"type": "null"}],
                "default": None,
                "description": "Search params for when there is no prefetch",
            },
            "score_threshold": {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "default": None,
                "description": "Return points with scores better than this threshold.",
                "title": "Score Threshold",
            },
            "limit": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Max number of points to return. Default is 10.",
                "title": "Limit",
            },
            "offset": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Offset of the result. Skip this many points. Default is 0",
                "title": "Offset",
            },
            "with_vector": {
                "anyOf": [
                    {"type": "boolean"},
                    {"items": {"type": "string"}, "type": "array"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Options for specifying which vectors to include into the response. Default is false.",
                "title": "With Vector",
            },
            "with_payload": {
                "anyOf": [
                    {"type": "boolean"},
                    {"items": {"type": "string"}, "type": "array"},
                    {"$ref": "#/$defs/PayloadSelectorInclude"},
                    {"$ref": "#/$defs/PayloadSelectorExclude"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Options for specifying which payload to include or not. Default is false.",
                "title": "With Payload",
            },
            "lookup_from": {
                "anyOf": [{"$ref": "#/$defs/LookupLocation"}, {"type": "null"}],
                "default": None,
                "description": "The location to use for IDs lookup, if not specified - use the current collection and the &#x27;using&#x27; vector Note: the other collection vectors should have the same vector size as the &#x27;using&#x27; vector in the current collection",
            },
        },
        "title": "QueryRequest",
        "type": "object",
    },
    "RecommendRequest": {
        "additionalProperties": False,
        "description": "Recommendation request. Provides positive and negative examples of the vectors, which can be ids of points that are already stored in the collection, raw vectors, or even ids and vectors combined.  Service should look for the points which are closer to positive examples and at the same time further to negative examples. The concrete way of how to compare negative and positive distances is up to the `strategy` chosen.",
        "properties": {
            "shard_key": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "Specify in which shards to look for the points, if not specified - look in all shards",
                "title": "Shard Key",
            },
            "positive": {
                "anyOf": [
                    {
                        "items": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "string"},
                                {"items": {"type": "number"}, "type": "array"},
                                {"$ref": "#/$defs/SparseVector"},
                            ]
                        },
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": [],
                "description": "Look for vectors closest to those",
                "title": "Positive",
            },
            "negative": {
                "anyOf": [
                    {
                        "items": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "string"},
                                {"items": {"type": "number"}, "type": "array"},
                                {"$ref": "#/$defs/SparseVector"},
                            ]
                        },
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": [],
                "description": "Try to avoid vectors like this",
                "title": "Negative",
            },
            "strategy": {
                "anyOf": [{"$ref": "#/$defs/RecommendStrategy"}, {"type": "null"}],
                "default": None,
                "description": "How to use positive and negative examples to find the results",
            },
            "filter": {
                "anyOf": [{"$ref": "#/$defs/Filter"}, {"type": "null"}],
                "default": None,
                "description": "Look only for points which satisfies this conditions",
            },
            "params": {
                "anyOf": [{"$ref": "#/$defs/SearchParams"}, {"type": "null"}],
                "default": None,
                "description": "Additional search params",
            },
            "limit": {
                "description": "Max number of result to return",
                "title": "Limit",
                "type": "integer",
            },
            "offset": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Offset of the first result to return. May be used to paginate results. Note: large offset values may cause performance issues.",
                "title": "Offset",
            },
            "with_payload": {
                "anyOf": [
                    {"type": "boolean"},
                    {"items": {"type": "string"}, "type": "array"},
                    {"$ref": "#/$defs/PayloadSelectorInclude"},
                    {"$ref": "#/$defs/PayloadSelectorExclude"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Select which payload to return with the response. Default is false.",
                "title": "With Payload",
            },
            "with_vector": {
                "anyOf": [
                    {"type": "boolean"},
                    {"items": {"type": "string"}, "type": "array"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Options for specifying which vectors to include into response. Default is false.",
                "title": "With Vector",
            },
            "score_threshold": {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "default": None,
                "description": "Define a minimal score threshold for the result. If defined, less similar results will not be returned. Score of the returned result might be higher or smaller than the threshold depending on the Distance function used. E.g. for cosine similarity only higher scores will be returned.",
                "title": "Score Threshold",
            },
            "using": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "description": "Define which vector to use for recommendation, if not specified - try to use default vector",
                "title": "Using",
            },
            "lookup_from": {
                "anyOf": [{"$ref": "#/$defs/LookupLocation"}, {"type": "null"}],
                "default": None,
                "description": "The location used to lookup vectors. If not specified - use current collection. Note: the other collection should have the same vector size as the current collection",
            },
        },
        "required": ["limit"],
        "title": "RecommendRequest",
        "type": "object",
    },
    "ReplicateShard": {
        "additionalProperties": False,
        "properties": {
            "shard_id": {"description": "", "title": "Shard Id", "type": "integer"},
            "to_peer_id": {"description": "", "title": "To Peer Id", "type": "integer"},
            "from_peer_id": {"description": "", "title": "From Peer Id", "type": "integer"},
            "method": {
                "anyOf": [
                    {"$ref": "#/$defs/ShardTransferMethodOneOf"},
                    {"$ref": "#/$defs/ShardTransferMethodOneOf1"},
                    {"$ref": "#/$defs/ShardTransferMethodOneOf2"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Method for transferring the shard from one node to another",
                "title": "Method",
            },
        },
        "required": ["shard_id", "to_peer_id", "from_peer_id"],
        "title": "ReplicateShard",
        "type": "object",
    },
    "RestartTransfer": {
        "additionalProperties": False,
        "properties": {
            "shard_id": {"description": "", "title": "Shard Id", "type": "integer"},
            "from_peer_id": {"description": "", "title": "From Peer Id", "type": "integer"},
            "to_peer_id": {"description": "", "title": "To Peer Id", "type": "integer"},
            "method": {
                "anyOf": [
                    {"$ref": "#/$defs/ShardTransferMethodOneOf"},
                    {"$ref": "#/$defs/ShardTransferMethodOneOf1"},
                    {"$ref": "#/$defs/ShardTransferMethodOneOf2"},
                ],
                "description": "",
                "title": "Method",
            },
        },
        "required": ["shard_id", "from_peer_id", "to_peer_id", "method"],
        "title": "RestartTransfer",
        "type": "object",
    },
    "NamedSparseVector": {
        "additionalProperties": False,
        "description": "Sparse vector data with name",
        "properties": {
            "name": {"description": "Name of vector data", "title": "Name", "type": "string"},
            "vector": {
                "allOf": [{"$ref": "#/$defs/SparseVector"}],
                "description": "Sparse vector data with name",
            },
        },
        "required": ["name", "vector"],
        "title": "NamedSparseVector",
        "type": "object",
    },
    "NamedVector": {
        "additionalProperties": False,
        "description": "Dense vector data with name",
        "properties": {
            "name": {"description": "Name of vector data", "title": "Name", "type": "string"},
            "vector": {
                "description": "Vector data",
                "items": {"type": "number"},
                "title": "Vector",
                "type": "array",
            },
        },
        "required": ["name", "vector"],
        "title": "NamedVector",
        "type": "object",
    },
    "SearchRequest": {
        "additionalProperties": False,
        "description": "Search request. Holds all conditions and parameters for the search of most similar points by vector similarity given the filtering restrictions.",
        "properties": {
            "shard_key": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "Specify in which shards to look for the points, if not specified - look in all shards",
                "title": "Shard Key",
            },
            "vector": {
                "anyOf": [
                    {"items": {"type": "number"}, "type": "array"},
                    {"$ref": "#/$defs/NamedVector"},
                    {"$ref": "#/$defs/NamedSparseVector"},
                ],
                "description": "Search request. Holds all conditions and parameters for the search of most similar points by vector similarity given the filtering restrictions.",
                "title": "Vector",
            },
            "filter": {
                "anyOf": [{"$ref": "#/$defs/Filter"}, {"type": "null"}],
                "default": None,
                "description": "Look only for points which satisfies this conditions",
            },
            "params": {
                "anyOf": [{"$ref": "#/$defs/SearchParams"}, {"type": "null"}],
                "default": None,
                "description": "Additional search params",
            },
            "limit": {
                "description": "Max number of result to return",
                "title": "Limit",
                "type": "integer",
            },
            "offset": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Offset of the first result to return. May be used to paginate results. Note: large offset values may cause performance issues.",
                "title": "Offset",
            },
            "with_payload": {
                "anyOf": [
                    {"type": "boolean"},
                    {"items": {"type": "string"}, "type": "array"},
                    {"$ref": "#/$defs/PayloadSelectorInclude"},
                    {"$ref": "#/$defs/PayloadSelectorExclude"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Select which payload to return with the response. Default is false.",
                "title": "With Payload",
            },
            "with_vector": {
                "anyOf": [
                    {"type": "boolean"},
                    {"items": {"type": "string"}, "type": "array"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Options for specifying which vectors to include into response. Default is false.",
                "title": "With Vector",
            },
            "score_threshold": {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "default": None,
                "description": "Define a minimal score threshold for the result. If defined, less similar results will not be returned. Score of the returned result might be higher or smaller than the threshold depending on the Distance function used. E.g. for cosine similarity only higher scores will be returned.",
                "title": "Score Threshold",
            },
        },
        "required": ["vector", "limit"],
        "title": "SearchRequest",
        "type": "object",
    },
    "SnapshotPriority": {
        "description": "Defines source of truth for snapshot recovery: `NoSync` means - restore snapshot without *any* additional synchronization. `Snapshot` means - prefer snapshot data over the current state. `Replica` means - prefer existing data over the snapshot.",
        "enum": ["no_sync", "snapshot", "replica"],
        "title": "SnapshotPriority",
        "type": "string",
    },
    "CollectionParamsDiff": {
        "additionalProperties": False,
        "properties": {
            "replication_factor": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Number of replicas for each shard",
                "title": "Replication Factor",
            },
            "write_consistency_factor": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Minimal number successful responses from replicas to consider operation successful",
                "title": "Write Consistency Factor",
            },
            "read_fan_out_factor": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "Fan-out every read request to these many additional remote nodes (and return first available response)",
                "title": "Read Fan Out Factor",
            },
            "on_disk_payload": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true - point&#x27;s payload will not be stored in memory. It will be read from the disk every time it is requested. This setting saves RAM by (slightly) increasing the response time. Note: those payload values that are involved in filtering and are indexed - remain in RAM.",
                "title": "On Disk Payload",
            },
        },
        "title": "CollectionParamsDiff",
        "type": "object",
    },
    "Disabled": {"const": "Disabled", "enum": ["Disabled"], "title": "Disabled", "type": "string"},
    "VectorParamsDiff": {
        "additionalProperties": False,
        "properties": {
            "hnsw_config": {
                "anyOf": [{"$ref": "#/$defs/HnswConfigDiff"}, {"type": "null"}],
                "default": None,
                "description": "Update params for HNSW index. If empty object - it will be unset.",
            },
            "quantization_config": {
                "anyOf": [
                    {"$ref": "#/$defs/ScalarQuantization"},
                    {"$ref": "#/$defs/ProductQuantization"},
                    {"$ref": "#/$defs/BinaryQuantization"},
                    {"$ref": "#/$defs/Disabled"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Update params for quantization. If none - it is left unchanged.",
                "title": "Quantization Config",
            },
            "on_disk": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "If true, vectors are served from disk, improving RAM usage at the cost of latency",
                "title": "On Disk",
            },
        },
        "title": "VectorParamsDiff",
        "type": "object",
    },
    "ClearPayloadOperation": {
        "additionalProperties": False,
        "properties": {
            "clear_payload": {
                "anyOf": [{"$ref": "#/$defs/PointIdsList"}, {"$ref": "#/$defs/FilterSelector"}],
                "description": "",
                "title": "Clear Payload",
            }
        },
        "required": ["clear_payload"],
        "title": "ClearPayloadOperation",
        "type": "object",
    },
    "DeleteOperation": {
        "additionalProperties": False,
        "properties": {
            "delete": {
                "anyOf": [{"$ref": "#/$defs/PointIdsList"}, {"$ref": "#/$defs/FilterSelector"}],
                "description": "",
                "title": "Delete",
            }
        },
        "required": ["delete"],
        "title": "DeleteOperation",
        "type": "object",
    },
    "DeletePayloadOperation": {
        "additionalProperties": False,
        "properties": {
            "delete_payload": {"allOf": [{"$ref": "#/$defs/DeletePayload"}], "description": ""}
        },
        "required": ["delete_payload"],
        "title": "DeletePayloadOperation",
        "type": "object",
    },
    "DeleteVectorsOperation": {
        "additionalProperties": False,
        "properties": {
            "delete_vectors": {"allOf": [{"$ref": "#/$defs/DeleteVectors"}], "description": ""}
        },
        "required": ["delete_vectors"],
        "title": "DeleteVectorsOperation",
        "type": "object",
    },
    "OverwritePayloadOperation": {
        "additionalProperties": False,
        "properties": {
            "overwrite_payload": {"allOf": [{"$ref": "#/$defs/SetPayload"}], "description": ""}
        },
        "required": ["overwrite_payload"],
        "title": "OverwritePayloadOperation",
        "type": "object",
    },
    "PointVectors": {
        "additionalProperties": False,
        "properties": {
            "id": {
                "anyOf": [{"type": "integer"}, {"type": "string"}],
                "description": "",
                "title": "Id",
            },
            "vector": {
                "anyOf": [
                    {"items": {"type": "number"}, "type": "array"},
                    {"items": {"items": {"type": "number"}, "type": "array"}, "type": "array"},
                    {
                        "additionalProperties": {
                            "anyOf": [
                                {"items": {"type": "number"}, "type": "array"},
                                {"$ref": "#/$defs/SparseVector"},
                                {
                                    "items": {"items": {"type": "number"}, "type": "array"},
                                    "type": "array",
                                },
                                {"$ref": "#/$defs/Document"},
                            ]
                        },
                        "type": "object",
                    },
                    {"$ref": "#/$defs/Document"},
                ],
                "description": "",
                "title": "Vector",
            },
        },
        "required": ["id", "vector"],
        "title": "PointVectors",
        "type": "object",
    },
    "PointsBatch": {
        "additionalProperties": False,
        "properties": {
            "batch": {"allOf": [{"$ref": "#/$defs/Batch"}], "description": ""},
            "shard_key": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "",
                "title": "Shard Key",
            },
        },
        "required": ["batch"],
        "title": "PointsBatch",
        "type": "object",
    },
    "PointsList": {
        "additionalProperties": False,
        "properties": {
            "points": {
                "description": "",
                "items": {"$ref": "#/$defs/PointStruct"},
                "title": "Points",
                "type": "array",
            },
            "shard_key": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "",
                "title": "Shard Key",
            },
        },
        "required": ["points"],
        "title": "PointsList",
        "type": "object",
    },
    "SetPayloadOperation": {
        "additionalProperties": False,
        "properties": {
            "set_payload": {"allOf": [{"$ref": "#/$defs/SetPayload"}], "description": ""}
        },
        "required": ["set_payload"],
        "title": "SetPayloadOperation",
        "type": "object",
    },
    "UpdateVectors": {
        "additionalProperties": False,
        "properties": {
            "points": {
                "description": "Points with named vectors",
                "items": {"$ref": "#/$defs/PointVectors"},
                "title": "Points",
                "type": "array",
            },
            "shard_key": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "type": "array",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": "",
                "title": "Shard Key",
            },
        },
        "required": ["points"],
        "title": "UpdateVectors",
        "type": "object",
    },
    "UpdateVectorsOperation": {
        "additionalProperties": False,
        "properties": {
            "update_vectors": {"allOf": [{"$ref": "#/$defs/UpdateVectors"}], "description": ""}
        },
        "required": ["update_vectors"],
        "title": "UpdateVectorsOperation",
        "type": "object",
    },
    "UpsertOperation": {
        "additionalProperties": False,
        "properties": {
            "upsert": {
                "anyOf": [{"$ref": "#/$defs/PointsBatch"}, {"$ref": "#/$defs/PointsList"}],
                "description": "",
                "title": "Upsert",
            }
        },
        "required": ["upsert"],
        "title": "UpsertOperation",
        "type": "object",
    },
    "ReshardingDirectionOneOf": {
        "description": "Scale up, add a new shard",
        "enum": ["up"],
        "title": "ReshardingDirectionOneOf",
        "type": "string",
    },
    "ReshardingDirectionOneOf1": {
        "description": "Scale down, remove a shard",
        "enum": ["down"],
        "title": "ReshardingDirectionOneOf1",
        "type": "string",
    },
    "StartResharding": {
        "additionalProperties": False,
        "properties": {
            "direction": {
                "anyOf": [
                    {"$ref": "#/$defs/ReshardingDirectionOneOf"},
                    {"$ref": "#/$defs/ReshardingDirectionOneOf1"},
                ],
                "description": "",
                "title": "Direction",
            },
            "peer_id": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "",
                "title": "Peer Id",
            },
            "shard_key": {
                "anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}],
                "default": None,
                "description": "",
                "title": "Shard Key",
            },
        },
        "required": ["direction"],
        "title": "StartResharding",
        "type": "object",
    },
}
RECURSIVE_REFS = ["Filter", "MinShould", "Nested", "NestedCondition", "Prefetch"]
INCLUDED_RECURSIVE_REFS = ["Prefetch"]
EXCLUDED_RECURSIVE_REFS = ["Filter", "MinShould", "Nested", "NestedCondition"]
NAME_RECURSIVE_REF_MAPPING = {"nested": "Nested", "prefetch": "Prefetch"}
