"""A caching utility for functions returning Pydantic models, optimized for deterministic and generative AI workflows.

This module exposes a single public decorator:

- `pydantic_cache`: Caches the result of a function returning a Pydantic model to a JSON file in
    the user's cache directory. The cache key is based on the function's arguments and a SHA-256
    hash of its source code (if available). Supports both forms:

    >>> @pydantic_cache
    ... def func(...): ...

    >>> @pydantic_cache()
    ... def func(...): ...

Internal implementation details are encapsulated in private classes and functions:

- `_FunctionCall`: Captures a function's name, arguments, and source code hash.
- `_CachedModel`: Stores a `_FunctionCall` and its corresponding result.
- `_get_signature()`: Generates a `_FunctionCall` from a function call.

Applications for Generative AI:

This decorator is especially useful when working with tools like **Pydantic AI** or **Magnetic**:

1. Avoid redundant calls to AI models when the same inputs are used repeatedly.
2. Improve reproducibility by linking cache entries to both function arguments and implementation.
3. Reduce latency and cost when experimenting with AI-powered pipelines that return structured Pydantic models.

Cache files are stored in the standard user cache directory (via `platformdirs`) under a
package-specific subdirectory, determined by the registered namespace. If multiple registered namespaces are applicable
(for example if a decorated function is in `top_library.subpackage`), then the most specific namespace is used.

Intended use:

- Deterministic functions returning Pydantic models.
- Scenarios where caching based on input arguments and function code is meaningful.
- Workflows involving structured AI outputs where repeated calls may be expensive.
"""

import functools
import hashlib
import inspect
import dataclasses
import threading
import pathlib
import warnings
from collections.abc import Callable
from typing import Any, Generic, TypeVar, cast, overload

import platformdirs
import pydantic
from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R", bound=pydantic.BaseModel)

@dataclasses.dataclass
class Namespace:
    package_name: str
    package_author: str | None

_NAMESPACES: dict[str, Namespace] = {}
_lock = threading.RLock()

def set_namespace(package_name: str, package_author: str | None = None, for_package: str | None = None) -> None:
    if for_package is None:
        frm = inspect.stack()[1].frame
        mod = inspect.getmodule(frm)
        if not mod:
            raise RuntimeError("cannot determine caller package")
        for_package = mod.__name__
    with _lock:
        _NAMESPACES[for_package] = Namespace(package_name, package_author)

def _resolve_namespace(module_name: str) -> Namespace:
    parts = module_name.split(".")
    with _lock:
        for i in range(len(parts), 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in _NAMESPACES:
                return _NAMESPACES[prefix]
    return Namespace(parts[0], None)





class _FunctionCall(pydantic.BaseModel):
    """Represents the details of a function call for caching purposes.

    Warning:
        This model is intended for internal use only and should not be instantiated directly by end users.

    """

    function_name: str
    """The string representation of the function being called."""
    function_arguments: dict[str, Any]
    """A dictionary of all positional and keyword arguments bound to the function call."""
    function_source_code_hash: str | None
    """SHA-256 hash of the function's source code at the time of definition. Used to invalidate cache if the function
    implementation changes. None if the source cannot be retrieved.
    """

    @property
    def hash_key(self) -> str:
        """SHA-256 hash of the serialized function call, used as a unique cache key."""
        json_dump = self.model_dump_json()
        return hashlib.sha256(json_dump.encode()).hexdigest()

    @property
    def file_name(self) -> str:
        """The cache file name derived from the hash key, ending in `.json`."""
        return self.hash_key + ".json"


class _CachedModel(pydantic.BaseModel, Generic[R]):
    """Represents a cached result of a function call, linking the function call to its output.

    This model is generic over `R`, the type of the result model, and is intended for internal use by the
    `pydantic_cache` decorator to persist and retrieve function outputs.

    Warning:
        This model is intended for internal use only and should not be instantiated directly by end users.

    """

    function_call: _FunctionCall
    """The function call details used to generate or retrieve this cache entry."""
    result: R
    """The output of the function, which must be a Pydantic model."""


def _get_signature(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> tuple[_FunctionCall, Any]:
    """Get the function call details, to be serialized alongside the model.

    Args:
        func: Function called.
        *args: Positional arguments to the function.
        **kwargs: Keyword arguments to the function.

    Return:
        Struct containing relevant details for recreating the function call.

    """
    signature = inspect.signature(func)
    bound_signature = signature.bind(*args, **kwargs)
    bound_signature.apply_defaults()
    call_args = dict(bound_signature.arguments)

    source_code_hash: str | None = None
    try:
        source_code = inspect.getsource(func)
    except OSError as e:
        warnings.warn(f"Could not retrieve source code. Skipping function source code hash: {e}", stacklevel=2)
    else:
        source_code_hash = hashlib.sha256(source_code.encode()).hexdigest()

    return (
        _FunctionCall(
            function_name=func.__qualname__, function_arguments=call_args, function_source_code_hash=source_code_hash
        ),
        signature.return_annotation,
    )


@overload
def pydantic_cache(func: None = None) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
@overload
def pydantic_cache(func: Callable[P, R]) -> Callable[P, R]: ...
def pydantic_cache(func: Callable[P, R] | None = None) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Cache the results of functions returning Pydantic models.

    The decorator serializes the function's input arguments and a SHA-256 hash of the function's
    source code to determine a unique cache key. Results are stored as JSON files in the user's
    cache directory (determined via `platformdirs`). Subsequent calls with the same arguments
    and unchanged function code will return the cached result, avoiding recomputation.

    Supports usage both with and without parentheses:

    >>> @pydantic_cache
    ... def func(...): ...

    >>> @pydantic_cache()
    ... def func(...): ...

    Args:
        func: The function to wrap. If None, returns a decorator that can be applied to a function later.

    Returns:
        A wrapped function that checks the cache before executing and stores the result if not already cached.

    Examples:
        Basic caching of a Pydantic model:

        >>> from pydantic import BaseModel
        >>> from your_module import pydantic_cache
        >>>
        >>> class ResultModel(BaseModel):
        ...     value: int
        >>>
        >>> @pydantic_cache
        ... def compute_square(x: int) -> ResultModel:
        ...     print("Computing...")
        ...     return ResultModel(value=x * x)
        >>>
        >>> compute_square(3)  # first call, prints "Computing..."
        Computing...
        ResultModel(value=9)
        >>> compute_square(3)  # second call, fetched from cache, no print
        ResultModel(value=9)

        Using parentheses:

        >>> @pydantic_cache()
        ... def compute_sum(a: int, b: int) -> ResultModel:
        ...     return ResultModel(value=a + b)
        >>> compute_sum(2, 4)
        ResultModel(value=6)

        Generative AI use case:

        >>> from pydantic import BaseModel
        >>> class AIResponse(BaseModel):
        ...     text: str
        >>>
        >>> @pydantic_cache
        ... def generate_text(prompt: str) -> AIResponse:
        ...     # Imagine an expensive AI call here
        ...     return AIResponse(text=f"Generated response for: {prompt}")
        >>>
        >>> generate_text("Hello, world!")  # first call, computes result
        AIResponse(text='Generated response for: Hello, world!')
        >>> generate_text("Hello, world!")  # second call, uses cache
        AIResponse(text='Generated response for: Hello, world!')

        Example using the OpenAI Python SDK with Pydantic:

        >>> from pydantic import BaseModel
        >>> from openai import OpenAI
        >>> class OpenAIResponse(BaseModel):
        ...     content: str
        >>>
        >>> @pydantic_cache
        ... def ask_openai(prompt: str) -> OpenAIResponse:
        ...     client = OpenAI()
        ...     response = client.chat.completions.create(
        ...         model="gpt-4",
        ...         messages=[{"role": "user", "content": prompt}]
        ...     )
        ...     return OpenAIResponse(content=response.choices[0].message.content)
        >>>
        >>> ask_openai("What is the capital of France?")
        OpenAIResponse(content='Answer to: What is the capital of France?')
        >>> ask_openai("What is the capital of France?")  # uses cache
        OpenAIResponse(content='Answer to: What is the capital of France?')

    Note:
        - Cache entries are invalidated automatically if the function's source code changes.
        - Works best with deterministic functions returning Pydantic models.
        - Cache files are stored under a package-specific directory in the user's cache folder, determined by
        `get_package_name()` and `get_package_author()`.
        - Internal classes `_FunctionCall` and `_CachedModel` handle serialization and should not be used directly by
        end users.

    """

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            function_call, return_type = _get_signature(f, *args, **kwargs)
            module = inspect.getmodule(f)
            if module is None or not module.__name__:
                namespace = Namespace("auto_pydantic_cache_default", None)
            else:
                namespace = _resolve_namespace(module.__name__)
            app_dirs = platformdirs.AppDirs(namespace.package_name, namespace.package_author)
            cache_dir = pathlib.Path(app_dirs.user_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = (cache_dir) / function_call.file_name
            if cache_file.exists():
                cached_model: _CachedModel[R] = cast(
                    "_CachedModel[R]", _CachedModel[return_type].model_validate_json(cache_file.read_text())
                )
                return cached_model.result

            result = f(*args, **kwargs)
            cached_model = _CachedModel(function_call=function_call, result=result)
            cache_file.write_text(cached_model.model_dump_json())
            return result

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
