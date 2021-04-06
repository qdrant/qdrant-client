# CollectionUpdateOperations

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**upsert_points** | [**PointInsertOperations**](PointInsertOperations.md) |  | [optional] 
**delete_points** | [**PointOperationsAnyOf1DeletePoints**](PointOperationsAnyOf1DeletePoints.md) |  | [optional] 
**set_payload** | [**PayloadOpsAnyOfSetPayload**](PayloadOpsAnyOfSetPayload.md) |  | [optional] 
**delete_payload** | [**PayloadOpsAnyOf1DeletePayload**](PayloadOpsAnyOf1DeletePayload.md) |  | [optional] 
**clear_payload** | [**PayloadOpsAnyOf2ClearPayload**](PayloadOpsAnyOf2ClearPayload.md) |  | [optional] 
**create_index** | **str** |  | [optional] 
**delete_index** | **str** |  | [optional] 
**any string name** | **bool, date, datetime, dict, float, int, list, str, none_type** | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


