from pydantic.version import VERSION as pydantic_version

PYDANTIC_V2 = pydantic_version.startswith("2.")


def update_forward_refs(model_class, *args, **kwargs):
    if PYDANTIC_V2:
        model_class.model_rebuild(*args, **kwargs)
    else:
        model_class.update_forward_refs(*args, **kwargs)


def construct(model, *args, **kwargs):
    if PYDANTIC_V2:
        return model.model_construct(*args, **kwargs)
    else:
        return model.construct(*args, **kwargs)


def to_dict(model, *args, **kwargs):
    if PYDANTIC_V2:
        return model.model_dump(*args, **kwargs)
    else:
        return model.dict(*args, **kwargs)


def to_json(model, *args, **kwargs):
    if PYDANTIC_V2:
        return model.model_dump(*args, **kwargs)
    else:
        return model.model_dump_json(*args, **kwargs)
