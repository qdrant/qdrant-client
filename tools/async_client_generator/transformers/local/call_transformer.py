import ast

from tools.async_client_generator.transformers.call_transformer import CallTransformer

# LocalCollection methods that run a synchronous, potentially long-running numpy scan
# with no internal await point. Left as plain calls, they'd monopolize the event loop
# for their full duration; asyncio.to_thread moves that work off the loop.
TO_THREAD_METHODS = frozenset(
    {
        "query_points",
        "query_groups",
        "scroll",
        "count",
        "facet",
        "search_matrix_offsets",
        "search_matrix_pairs",
        "retrieve",
    }
)


class LocalCallTransformer(CallTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST | ast.Await:
        if isinstance(node.func, ast.Name):
            if node.func.id in self.class_replace_map:
                node.func.id = self.class_replace_map[node.func.id]

        if isinstance(node.func, ast.Attribute):
            if node.func.attr in self.async_methods:
                if getattr(node.func.value, "id", None) == "self":
                    return ast.Await(value=node)

            if node.func.attr in TO_THREAD_METHODS:
                if getattr(node.func.value, "id", None) != "self":
                    to_thread_call = ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="asyncio", ctx=ast.Load()),
                            attr="to_thread",
                            ctx=ast.Load(),
                        ),
                        args=[node.func, *node.args],
                        keywords=node.keywords,
                    )
                    ast.copy_location(to_thread_call, node)
                    await_node = ast.Await(value=to_thread_call)
                    ast.copy_location(await_node, node)
                    return await_node

        return self.generic_visit(node)
