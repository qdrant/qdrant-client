# OptimizersConfigDiff

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**deleted_threshold** | **float, none_type** | The minimal fraction of deleted vectors in a segment, required to perform segment optimization | [optional] 
**flush_interval_sec** | **int, none_type** | Minimum interval between forced flushes. | [optional] 
**indexing_threshold** | **int, none_type** | Maximum number of vectors allowed for plain index. Default value based on https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md | [optional] 
**max_segment_number** | **int, none_type** | If the number of segments exceeds this value, the optimizer will merge the smallest segments. | [optional] 
**memmap_threshold** | **int, none_type** | Maximum number of vectors to store in-memory per segment. Segments larger than this threshold will be stored as read-only memmaped file. | [optional] 
**payload_indexing_threshold** | **int, none_type** | Starting from this amount of vectors per-segment the engine will start building index for payload. | [optional] 
**vacuum_min_vector_number** | **int, none_type** | The minimal number of vectors in a segment, required to perform segment optimization | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


