# OptimizersConfig

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**deleted_threshold** | **float** | The minimal fraction of deleted vectors in a segment, required to perform segment optimization | 
**flush_interval_sec** | **int** | Minimum interval between forced flushes. | 
**indexing_threshold** | **int** | Maximum number of vectors allowed for plain index. Default value based on https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md | 
**max_segment_number** | **int** | If the number of segments exceeds this value, the optimizer will merge the smallest segments. | 
**memmap_threshold** | **int** | Maximum number of vectors to store in-memory per segment. Segments larger than this threshold will be stored as read-only memmaped file. | 
**payload_indexing_threshold** | **int** | Starting from this amount of vectors per-segment the engine will start building index for payload. | 
**vacuum_min_vector_number** | **int** | The minimal number of vectors in a segment, required to perform segment optimization | 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


