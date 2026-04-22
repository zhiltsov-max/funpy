import itertools
import operator
from collections.abc import Callable, Sequence
from functools import partial, wraps
from typing import Any, TypeVar, cast

from funpy.context import (
    CallArgs,
    CallContext,
    ContextRef,
    GenericContextRef,
    NamedArgRef,
    PositionalArgRef,
    PostCallContext,
    PreCallContext,
    ResultRef,
)
from funpy.errors import InvalidContextReferenceError
from funpy.sentinel import (
    ContextRefSentinel,
    _input_passthrough,
    _rv,
    is_context_ref_sentinel,
)


def apply_ref(ref: Any | ContextRef | ContextRefSentinel, *, context: CallContext) -> Any:
    if is_context_ref_sentinel(ref):
        s = cast(ContextRefSentinel, ref)
        ref = s()  # TODO: try to implement type checked narrowing

    if isinstance(ref, ContextRef):
        if isinstance(ref, GenericContextRef):
            return ref.transform(context)

        transform = getattr(ref, "transform", None)
        if isinstance(ref, PositionalArgRef):
            v = context.args[ref.index]
        elif isinstance(ref, NamedArgRef):
            v = context.kwargs[ref.name]
        elif isinstance(ref, ResultRef):
            if isinstance(context, PreCallContext):
                raise InvalidContextReferenceError(
                    f"The context reference '{ref}' requires the wrapped function call result, "
                    f"so cannot be used in the context before the wrapped function called."
                )
            elif isinstance(context, PostCallContext):
                v = context.result
            else:
                raise AssertionError(f"Unknown call context type '{type(context).__name__}'")
        else:
            raise NotImplementedError

        if transform:
            v = transform(v)
    else:
        v = ref

    return v


R = TypeVar("R")
Tr = TypeVar("Tr")

_UNDEFINED = object()


def wrap(
    f: Callable,
    *args: ContextRef | ContextRefSentinel | Any,
    _return: ContextRef | ContextRefSentinel | Any = _rv,
    **kwargs: ContextRef | ContextRefSentinel | Any,
) -> Callable:
    """
    Modify the function to have different parameters or return values.

    Positional args used in the decorator are appended to the regular positionals during the call.

    Named args used in the decorator replace the named args during the call. Named args can also
    be specified as _<number> (e.g. wrap(f, _2="hello world")) to be used as positionals
    with the specified indices.

    The return value of the function can also be modified with the "_return" parameter.

    Sentinels (e.g. _0, _1, _rv, _kw["arg"]) are recommended for use wherever possible.
    """

    @wraps(f)
    def wrapped(*input_args, **input_kwargs):
        if len(input_args) == 1 and not input_kwargs and isinstance(input_args[0], CallArgs):
            input_args, input_kwargs = input_args[0].as_tuple()

        context = PreCallContext(func=f, args=input_args, kwargs=input_kwargs)

        call_args = [
            apply_ref(arg, context=context) for arg in itertools.chain(args or (), input_args)
        ]

        for k in tuple(kwargs):
            if k.startswith("_") and k[1:].isdigit():
                index = int(k[1:])

                arg = kwargs.pop(k)
                arg = apply_ref(arg, context=context)

                if len(call_args) <= index:
                    call_args.extend([_UNDEFINED] * (index + 1 - len(call_args)))

                call_args[index] = arg

        defined_arg_count = sum(v is not _UNDEFINED for v in call_args)
        if defined_arg_count != len(call_args):
            raise TypeError(
                "{} expected at least {} arguments, got {}".format(
                    f.__name__, len(call_args), defined_arg_count
                )
            )

        call_kwargs = dict(input_kwargs)
        call_kwargs.update(
            {k: apply_ref(arg, context=context) for k, arg in (kwargs or {}).items()}
        )

        call_args = tuple(call_args)

        call_result = f(*call_args, **call_kwargs)

        context = PostCallContext(func=f, args=call_args, kwargs=call_kwargs, result=call_result)
        return apply_ref(_return, context=context)

    return wrapped


side_call = partial(wrap, _return=_input_passthrough)


def printify(
    f: Callable[..., R],
    *,
    fmt: str,
    fmt_args: Sequence[Any | ContextRef] | None = None,
    fmt_kwargs: dict[str, Any | ContextRef] | None = None,
):
    def format_arg(arg: Any | ContextRef | ContextRefSentinel, *, context: PostCallContext) -> Any:
        return apply_ref(arg, context=context)

    @wraps(f)
    def printified(*args, **kwargs):
        result = f(*args, **kwargs)

        context = PostCallContext(func=f, args=args, kwargs=kwargs, result=result)
        format_args = [format_arg(arg, context=context) for arg in (fmt_args or ())]
        format_kwargs = {k: format_arg(kwarg, context=context) for k, kwarg in (fmt_kwargs or {})}
        print(fmt.format(*format_args, **format_kwargs))

        return result

    return printified


def thresholdify(f: Callable[..., R], *, threshold: Tr, op: Callable[[R, Tr], bool] = operator.ge):
    @wraps(f)
    def thresholded(*args, **kwargs):
        return op(f(*args, **kwargs), threshold)

    return thresholded


def chain(*funcs: Callable) -> Callable:
    def _chained(*input_args, **input_kwargs):
        call_args = CallArgs(args=input_args, kwargs=input_kwargs)
        for fn in funcs:
            result = fn(*call_args.args, **call_args.kwargs)

            if isinstance(result, CallArgs):
                call_args = result
            else:
                call_args = CallArgs.from_return(result)

        if call_args.is_trivial:
            return call_args.value()
        else:
            return call_args

    return _chained
