import asyncio

from grpclib.client import Channel

from qdrant_client.grpc import CollectionsStub


async def grpc_query():
    async with Channel(host="localhost", port=6334) as channel:
        service = CollectionsStub(channel)
        response = await service.list()
        print(response.collections)


def test_grpc_call():
    asyncio.run(grpc_query())

#
#
# if __name__ == "__main__":
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(grpc_query())
