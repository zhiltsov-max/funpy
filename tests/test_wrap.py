from typing import TypeVar

import funpy as fp

_T = TypeVar("_T")


def binary_identity(a: _T, b: _T) -> tuple[_T, _T]:
    return a, b


def test_can_bind_positionals():
    wrapped = fp.wrap(binary_identity, _0=fp._1, _1=fp._0)

    assert wrapped(0, 1) == (1, 0)


def test_can_bind_positionals_to_kwargs():
    wrapped = fp.wrap(binary_identity, _0=fp._kw["foo"], _1=fp._kw["bar"])

    assert wrapped(foo=1, bar=2) == (1, 2)


def test_can_bind_kwargs():
    wrapped = fp.wrap(binary_identity, a=42)

    assert wrapped(1) == (42, 1)


def test_can_bind_result():
    wrapped = fp.wrap(
        binary_identity,
        _return=fp.GenericContextRef(transform=lambda ctx: ctx.args[0] + ctx.args[1]),
    )

    assert wrapped(1, 2) == 3


def test_can_make_side_call():
    side_called_args = []

    def side_call(*args, **kwargs):
        nonlocal side_called_args
        side_called_args.append({"args": args, "kwargs": kwargs})

    wrapped = fp.side_call(side_call)

    result = wrapped("hello", kwarg="world")

    assert side_called_args == [
        {
            "args": ("hello",),
            "kwargs": {
                "kwarg": "world",
            },
        }
    ]

    assert isinstance(result, fp.CallArgs)
    assert result.args == ("hello",)
    assert result.kwargs == {"kwarg": "world"}
