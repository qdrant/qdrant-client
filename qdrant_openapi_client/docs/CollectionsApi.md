# qdrant_openapi_client.CollectionsApi

All URIs are relative to *http://localhost:6333*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_collection**](CollectionsApi.md#get_collection) | **GET** /collections/{name} | Get information about existing collection
[**get_collections**](CollectionsApi.md#get_collections) | **GET** /collections | Get list of existing collections
[**update_collections**](CollectionsApi.md#update_collections) | **POST** /collections | Perform update operation on collections


# **get_collection**
> InlineResponse2002 get_collection(name)

Get information about existing collection

### Example

```python
import time
import qdrant_openapi_client
from qdrant_openapi_client.api import collections_api
from qdrant_openapi_client.model.error_response import ErrorResponse
from qdrant_openapi_client.model.inline_response2002 import InlineResponse2002
from pprint import pprint
# Defining the host is optional and defaults to http://localhost:6333
# See configuration.py for a list of all supported configuration parameters.
configuration = qdrant_openapi_client.Configuration(
    host = "http://localhost:6333"
)


# Enter a context with an instance of the API client
with qdrant_openapi_client.ApiClient() as api_client:
    # Create an instance of the API class
    api_instance = collections_api.CollectionsApi(api_client)
    name = "name_example" # str | Name of the collection to retrieve

    # example passing only required values which don't have defaults set
    try:
        # Get information about existing collection
        api_response = api_instance.get_collection(name)
        pprint(api_response)
    except qdrant_openapi_client.ApiException as e:
        print("Exception when calling CollectionsApi->get_collection: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **name** | **str**| Name of the collection to retrieve |

### Return type

[**InlineResponse2002**](InlineResponse2002.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | successful operation |  -  |
**0** | error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_collections**
> InlineResponse200 get_collections()

Get list of existing collections

### Example

```python
import time
import qdrant_openapi_client
from qdrant_openapi_client.api import collections_api
from qdrant_openapi_client.model.inline_response200 import InlineResponse200
from qdrant_openapi_client.model.error_response import ErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to http://localhost:6333
# See configuration.py for a list of all supported configuration parameters.
configuration = qdrant_openapi_client.Configuration(
    host = "http://localhost:6333"
)


# Enter a context with an instance of the API client
with qdrant_openapi_client.ApiClient() as api_client:
    # Create an instance of the API class
    api_instance = collections_api.CollectionsApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        # Get list of existing collections
        api_response = api_instance.get_collections()
        pprint(api_response)
    except qdrant_openapi_client.ApiException as e:
        print("Exception when calling CollectionsApi->get_collections: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**InlineResponse200**](InlineResponse200.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | successful operation |  -  |
**0** | error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_collections**
> InlineResponse2001 update_collections()

Perform update operation on collections

### Example

```python
import time
import qdrant_openapi_client
from qdrant_openapi_client.api import collections_api
from qdrant_openapi_client.model.storage_operations import StorageOperations
from qdrant_openapi_client.model.inline_response2001 import InlineResponse2001
from qdrant_openapi_client.model.error_response import ErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to http://localhost:6333
# See configuration.py for a list of all supported configuration parameters.
configuration = qdrant_openapi_client.Configuration(
    host = "http://localhost:6333"
)


# Enter a context with an instance of the API client
with qdrant_openapi_client.ApiClient() as api_client:
    # Create an instance of the API class
    api_instance = collections_api.CollectionsApi(api_client)
    storage_operations = StorageOperations() # StorageOperations | Operation to perform on collections (optional)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Perform update operation on collections
        api_response = api_instance.update_collections(storage_operations=storage_operations)
        pprint(api_response)
    except qdrant_openapi_client.ApiException as e:
        print("Exception when calling CollectionsApi->update_collections: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **storage_operations** | [**StorageOperations**](StorageOperations.md)| Operation to perform on collections | [optional]

### Return type

[**InlineResponse2001**](InlineResponse2001.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | successful operation |  -  |
**0** | error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

