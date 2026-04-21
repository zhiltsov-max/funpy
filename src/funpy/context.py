from collections.abc import Sequence
from typing import Any, Callable

from attrs import define

from funpy.errors import CompositeCallArgsError


@define
class ContextRef:
    pass


@define
class CallContext:
    func: Callable
    args: Sequence[Any]
    kwargs: dict[str, Any]


@define
class PreCallContext(CallContext):
    pass


@define
class PostCallContext(CallContext):
    result: Any


@define
class GenericContextRef(ContextRef):
    transform: Callable[[CallContext], Any]


@define(kw_only=True)
class SimpleTransformableRef:
    transform: Callable[[Any], Any] | None = None


@define
class PositionalArgRef(ContextRef, SimpleTransformableRef):
    index: int


@define
class NamedArgRef(ContextRef, SimpleTransformableRef):
    name: str


@define
class ResultRef(ContextRef, SimpleTransformableRef):
    pass


@define
class CallArgs:
    args: Sequence[Any]
    kwargs: dict[str, Any]

    def __len__(self) -> int:
        if self.kwargs:
            raise CompositeCallArgsError()

        return len(self.args)

    def __getitem__(self, i: int | str) -> Any:
        if isinstance(i, int):
            return self.args[i]
        elif isinstance(i, str):
            return self.kwargs[i]
        else:
            raise TypeError

    def __bool__(self) -> bool:
        return self.args or self.kwargs

    def as_tuple(self) -> tuple[Sequence[Any], dict[str, Any]]:
        return self.args, self.kwargs
