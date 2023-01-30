from typing import Optional, List, Any

import grpc


def get_channel(host: str, port: int, ssl: bool, metadata: Optional[List[Any]] = None) -> grpc.Channel:
    if ssl:

        if metadata:
            def metadata_callback(context, callback):
                # for more info see grpc docs
                callback(metadata, None)

            # build ssl credentials using the cert the same as before
            cert_creds = grpc.ssl_channel_credentials()

            # now build meta data credentials
            auth_creds = grpc.metadata_call_credentials(metadata_callback)

            # combine the cert credentials and the macaroon auth credentials
            # such that every call is properly encrypted and authenticated
            creds = grpc.composite_channel_credentials(cert_creds, auth_creds)
        else:
            creds = grpc.ssl_channel_credentials()

        # finally pass in the combined credentials when creating a channel
        return grpc.secure_channel(f'{host}:{port}', creds)
    else:
        return grpc.insecure_channel(f'{host}:{port}')
