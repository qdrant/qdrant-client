from tests.congruence_tests.test_common import init_local, init_remote


def test_info():
    local_client = init_local()
    remote_client = init_remote()
    remote_grpc_client = init_remote(prefer_grpc=True)

    local_info = local_client.info()
    rest_info = remote_client.info()
    grpc_info = remote_grpc_client.info()

    assert local_info.title == rest_info.title
    assert local_info.version is not None
    assert rest_info.version == grpc_info.version
    assert rest_info == grpc_info
