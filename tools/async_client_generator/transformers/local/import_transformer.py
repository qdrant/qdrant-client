import ast


class LocalEnsureImportTransformer(ast.NodeTransformer):
    """Prepends `import asyncio`, needed by LocalCallTransformer's asyncio.to_thread wraps.

    The sync source this generator reads from has no reason to import asyncio itself, so
    the import can't come from the shared source -- it only makes sense in the async output.
    """

    def visit_Module(self, node: ast.Module) -> ast.AST:
        import_node = ast.Import(names=[ast.alias(name="asyncio", asname=None)])
        ast.copy_location(import_node, node.body[0])
        node.body.insert(0, import_node)
        return node
