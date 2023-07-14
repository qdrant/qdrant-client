from typing import Any, Dict, Type

from pydantic import BaseModel
from pydantic.version import VERSION as pydantic_version

PYDANTIC_V2 = pydantic_version.startswith("2.")


def update_forward_refs(model_class: Type[BaseModel], *args: Any, **kwargs: Any) -> None:
    if PYDANTIC_V2:
        model_class.model_rebuild(*args, **kwargs)
    else:
        model_class.update_forward_refs(*args, **kwargs)


def construct(model_class: Type[BaseModel], *args: Any, **kwargs: Any) -> BaseModel:
    if PYDANTIC_V2:
        return model_class.model_construct(*args, **kwargs)
    else:
        return model_class.construct(*args, **kwargs)


def to_dict(model: BaseModel, *args: Any, **kwargs: Any) -> Dict[Any, Any]:
    if PYDANTIC_V2:
        return model.model_dump(*args, **kwargs)
    else:
        return model.dict(*args, **kwargs)


def to_json(model: BaseModel, *args: Any, **kwargs: Any) -> str:
    if PYDANTIC_V2:
        return model.model_dump_json(*args, **kwargs)
    else:
        return model.json()
