import operator
from typing import TypeVar

from attrs import define, field

import funpy as fp

_T = TypeVar("_T")


def binary_identity(a: _T, b: _T) -> tuple[_T, _T]:
    return a, b


def test_can_bind_positionals():
    wrapped = fp.wrap(binary_identity, _0=fp._1, _1=fp._0)

    assert wrapped(0, 1) == (1, 0)


def test_can_bind_positionals_to_kwargs():
    wrapped = fp.wrap(binary_identity, _0=fp._kw("bar"), _1=fp._kw("foo"))

    assert wrapped(foo=1, bar=2) == (2, 1)


def test_can_bind_kwargs():
    wrapped = fp.wrap(binary_identity, b=42)

    assert wrapped(1) == (1, 42)


def test_can_bind_result():
    wrapped = fp.wrap(
        binary_identity,
        _return=fp.GenericContextRef(transform=lambda ctx: ctx.args[0] + ctx.args[1]),
    )

    assert wrapped(1, 2) == 3


@define
class CallRecorder:
    calls: list[fp.CallArgs] = field(factory=list)

    def __call__(self, *args, **kwargs):
        self.calls.append(fp.CallArgs(args=args, kwargs=kwargs))


def test_can_make_side_call():
    recorder = CallRecorder()

    wrapped = fp.side_call(recorder)

    result = wrapped("hello", kwarg="world")

    assert [
        fp.CallArgs(
            args=("hello",),
            kwargs={"kwarg": "world"},
        )
    ] == recorder.calls

    assert isinstance(result, fp.CallArgs)
    assert result.args == ("hello",)
    assert result.kwargs == {"kwarg": "world"}


def test_can_chain_calls():
    recorder = CallRecorder()

    def _ternary_func(a: int, b: int, c: int) -> int:
        return c

    chain = fp.chain(
        operator.add,
        fp.side_call(recorder),
        fp.wrap(operator.sub, _1=2),
        fp.side_call(recorder),
        fp.wrap(_ternary_func, _0=0, _1=1, _2=fp._0),
    )

    assert chain(1, 5) == 4
    assert recorder.calls == [fp.CallArgs(args=(6,)), fp.CallArgs(args=(4,))]


def test_can_pipe_calls():
    recorder = CallRecorder()

    def _ternary_func(a: int, b: int, c: int) -> int:
        return c

    pipe = (
        fp.wrap(operator.add)
        | fp.side_call(recorder)
        | fp.wrap(operator.sub, _1=2)
        | fp.side_call(recorder)
        | fp.wrap(_ternary_func, _0=0, _1=1, _2=fp._0)
        | operator.neg
    )

    assert pipe(1, 5) == -4
    assert recorder.calls == [fp.CallArgs(args=(6,)), fp.CallArgs(args=(4,))]
