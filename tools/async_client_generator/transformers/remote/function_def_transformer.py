import ast
from typing import List, Optional, Union

from tools.async_client_generator.transformers import FunctionDefTransformer


class RemoteFunctionDefTransformer(FunctionDefTransformer):
    def __init__(
        self,
        keep_sync: Optional[List[str]] = None,
        exclude_methods: Optional[List[str]] = None,
        async_methods: Optional[List[str]] = None,
    ):
        super().__init__(keep_sync=keep_sync)
        self.exclude_methods = exclude_methods if exclude_methods is not None else []
        self.async_methods = async_methods if async_methods is not None else []

    def _keep_sync(self, name: str) -> bool:
        return name in self.keep_sync or name not in self.async_methods

    @staticmethod
    def override_init(sync_node: ast.FunctionDef) -> ast.FunctionDef:
        # remove aio grpc and remain only pure grpc instead
        kick_assignments: List[Union[ast.Assign, ast.AnnAssign]] = []
        for child_node in sync_node.body:
            if isinstance(child_node, ast.Assign):
                for target in child_node.targets:
                    if isinstance(target, ast.Attribute) and hasattr(target, "attr"):
                        if "aio" in target.attr:
                            kick_assignments.append(child_node)
            if isinstance(child_node, ast.AnnAssign) and hasattr(child_node.target, "attr"):
                if "aio" in child_node.target.attr:
                    kick_assignments.append(child_node)
        sync_node.body = [node for node in sync_node.body if node not in kick_assignments]
        return sync_node

    @staticmethod
    def override_close() -> ast.stmt:
        code = """
async def close(self, grpc_grace: Optional[float] = None, **kwargs: Any) -> None:
    if hasattr(self, "_grpc_channel") and self._grpc_channel is not None:
        try:
            await self._grpc_channel.close(grace=grpc_grace)
        except AttributeError:
            logging.warning(
                "Unable to close grpc_channel. Connection was interrupted on the server side"
            )
        except RuntimeError:
            pass

    try:
        await self.http.aclose()
    except Exception:
        logging.warning(
            "Unable to close http connection. Connection was interrupted on the server side"
        )

    self._closed = True
            """

        parsed_code = ast.parse(code)
        return parsed_code.body[0]

    def visit_FunctionDef(self, sync_node: ast.FunctionDef) -> Optional[ast.AST]:
        if sync_node.name in self.exclude_methods:
            return None

        if sync_node.name == "__init__":
            sync_node = self.override_init(sync_node)
            return self.generic_visit(sync_node)

        if sync_node.name == "close":
            return self.override_close()

        return super().visit_FunctionDef(sync_node)
