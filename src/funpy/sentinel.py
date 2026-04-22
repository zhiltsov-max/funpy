from functools import partial
from typing import Any, Callable, Protocol

from funpy.context import (
    CallArgs,
    ContextRef,
    GenericContextRef,
    NamedArgRef,
    PositionalArgRef,
    ResultRef,
)

_0 = partial(PositionalArgRef, index=0)
_1 = partial(PositionalArgRef, index=1)
_2 = partial(PositionalArgRef, index=2)
_3 = partial(PositionalArgRef, index=3)
_4 = partial(PositionalArgRef, index=4)
_5 = partial(PositionalArgRef, index=5)
_6 = partial(PositionalArgRef, index=6)
_7 = partial(PositionalArgRef, index=7)
_8 = partial(PositionalArgRef, index=8)
_9 = partial(PositionalArgRef, index=9)
_kw = partial(NamedArgRef, consume=True)
_rv = partial(ResultRef)
_input_passthrough = partial(
    GenericContextRef, transform=lambda ctx: CallArgs(args=ctx.args, kwargs=ctx.kwargs)
)


class ContextRefSentinel(Protocol):
    __call__: Callable[[], ContextRef]


CONTEXT_REF_SENTINELS: set[int] = set()


def register_context_ref_sentinel(obj: Any):
    CONTEXT_REF_SENTINELS.add(id(obj))


def is_context_ref_sentinel(obj: Any) -> bool:
    return id(obj) in CONTEXT_REF_SENTINELS


for v in (_kw, _rv, _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _input_passthrough):
    register_context_ref_sentinel(v)
