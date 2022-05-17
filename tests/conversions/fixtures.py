from typing import List

import betterproto

from qdrant_client import grpc


def get_filters() -> List[grpc.Filter]:
    return [
        grpc.Filter(
            must=[
                grpc.Condition(
                    has_id=grpc.HasIdCondition(has_id=[
                        grpc.PointId(num=1),
                        grpc.PointId(num=2),
                        grpc.PointId(uuid="f9bcf279-5e66-40f7-856b-3a9d9b6617ee"),
                    ])
                )
            ]
        )
    ]


fixtures = {
    "CollectionParams": [...],
    "CollectionConfig": [...],
    "ScoredPoint": [...],
    "CreateAlias": [...],
    "GeoBoundingBox": [...],
    "SearchParams": [...],
    "HasIdCondition": [...],
    "RenameAlias": [...],
    "ValuesCount": [...],
    "Filter": get_filters(),
    "OptimizersConfigDiff": [...],
    "CollectionInfo": [...],
    "CreateCollection": [...],
    "FieldCondition": [...],
    "GeoRadius": [...],
    "UpdateResult": [...],
    "IsEmptyCondition": [...],
    "DeleteAlias": [...],
    "PointStruct": [...],
    "CollectionDescription": [...],
    "GeoPoint": [...],
    "WalConfigDiff": [...],
    "HnswConfigDiff": [...],
    "Range": [...],
    "UpdateCollection": [...],
    "Condition": [...],
    "PointsSelector": [...],
    "AliasOperations": [...]
}


def get_grpc_fixture(model_name: str) -> List[betterproto.Message]:
    if model_name not in fixtures:
        raise RuntimeError(f"Model {model_name} not fount in fixtures")
    return fixtures[model_name]
