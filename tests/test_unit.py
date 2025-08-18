import hashlib
import inspect
import json
import threading
from pathlib import Path
from typing import Generic, TypeVar

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_cache import _get_signature, pydantic_cache


@pytest.fixture
def tmp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Override package cache directory to a temporary path."""
    monkeypatch.setattr("pydantic_cache.get_package_name", lambda: "test_package")
    monkeypatch.setattr("pydantic_cache.get_package_author", lambda: "test_author")
    monkeypatch.setattr(
        "pydantic_cache.platformdirs.AppDirs",
        lambda *_args, **kwargs: type("AppDirs", (), {"user_cache_dir": tmp_path})(),  # noqa: ARG005
    )
    return tmp_path


class ResultModel(BaseModel):
    """Minimal Pydantic model with a single integer field, used to test caching of simple results."""

    value: int


class NestedModel(BaseModel):
    """Pydantic model containing another model as a field, used to test caching of nested structures."""

    inner: ResultModel


class AIResponse(BaseModel):
    """Pydantic model simulating an AI response with a single text field, used to test caching of generated outputs."""

    text: str


def test_cache_creates_file(tmp_cache_dir: Path) -> None:
    """Verify that calling a cached function creates a cache file on disk."""

    @pydantic_cache
    def f(x: int) -> ResultModel:
        return ResultModel(value=x)

    f(10)
    files = list(tmp_cache_dir.iterdir())
    assert len(files) == 1


def test_cache_returns_cached_result(tmp_cache_dir: Path) -> None:
    """Ensure repeated calls with the same arguments return the cached result and do not re-execute the function."""
    calls = []

    @pydantic_cache
    def f(x: int) -> ResultModel:
        calls.append(x)
        return ResultModel(value=x)

    f(1)
    f(1)
    assert calls == [1]  # Only first call executed


def test_different_args_create_different_cache_files(tmp_cache_dir: Path) -> None:
    """Check that different function arguments produce distinct cache files."""

    @pydantic_cache
    def f(x: int) -> ResultModel:
        return ResultModel(value=x)

    f(1)
    f(2)
    assert len(list(tmp_cache_dir.iterdir())) == 2


def test_cache_invalidated_on_source_change(tmp_cache_dir: Path) -> None:
    """Confirm that changing the function source code invalidates the old cache and creates a new file."""

    @pydantic_cache
    def f(x: int) -> ResultModel:
        return ResultModel(value=x)

    f(1)

    # Change function behavior
    @pydantic_cache
    def f(x: int) -> ResultModel:
        return ResultModel(value=x + 1)

    f(1)
    assert len(list(tmp_cache_dir.iterdir())) == 2


def test_function_with_no_source_code(tmp_cache_dir: Path) -> None:
    """Ensure functions without retrievable source code can still be cached and executed."""

    def f(x:int) -> ResultModel:
        return ResultModel(value=x)

    cached = pydantic_cache(f)
    result = cached(5)
    assert result.value == 5


def test_no_arguments(tmp_cache_dir: Path) -> None:
    """Verify that functions with no arguments are cached and return the correct result."""

    @pydantic_cache
    def f() -> ResultModel:
        return ResultModel(value=42)

    assert f().value == 42


def test_only_keyword_arguments(tmp_cache_dir: Path) -> None:
    """Ensure caching works for keyword-only arguments and argument order does not matter."""

    @pydantic_cache
    def f(x: int = 1, y: int = 2) -> ResultModel:
        return ResultModel(value=x + y)

    assert f(x=3, y=4).value == 7
    assert f(y=4, x=3).value == 7  # argument order irrelevant


def test_mutable_default_arguments(tmp_cache_dir: Path) -> None:
    """Check that mutable default arguments persist across calls and caching handles them correctly."""

    @pydantic_cache
    def f(lst: list | None = None) -> ResultModel:
        if lst is None:
            lst = []
        lst.append(1)
        return ResultModel(value=len(lst))

    listlist = []

    r1 = f(listlist)
    r2 = f(listlist)
    assert r1.value == 1
    assert r2.value == 2  # mutable default list persists across calls


def test_cache_dir_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm that the cache directory is created automatically and populated with files."""
    monkeypatch.setattr("pydantic_cache.get_package_name", lambda: "pkg")
    monkeypatch.setattr("pydantic_cache.get_package_author", lambda: "auth")
    monkeypatch.setattr(
        "pydantic_cache.platformdirs.AppDirs", lambda *_args: type("AppDirs", (), {"user_cache_dir": tmp_path})()
    )

    @pydantic_cache
    def f(x: int) -> ResultModel:
        return ResultModel(value=x)

    f(1)
    assert tmp_path.exists()
    assert any(tmp_path.iterdir())


def test_cache_serialization(tmp_cache_dir: Path) -> None:
    """Verify that cached results are serialized into JSON with the correct top-level keys."""

    @pydantic_cache
    def f(x: int) -> ResultModel:
        return ResultModel(value=x)

    f(5)
    file = next(tmp_cache_dir.iterdir())
    data = json.loads(file.read_text())
    assert "function_call" in data
    assert "result" in data


def test_nested_model(tmp_cache_dir: Path) -> None:
    """Ensure that nested Pydantic models are cached and loaded correctly."""

    @pydantic_cache
    def f(x: int) -> NestedModel:
        return NestedModel(inner=ResultModel(value=x))

    r1 = f(3)
    r2 = f(3)
    assert r1 == r2


def test_optional_fields(tmp_cache_dir: Path) -> None:
    """Check that models with optional fields are cached and returned as expected."""

    class OptionalModel(BaseModel):
        x: int | None = None

    @pydantic_cache
    def f(val: int | None = None) -> OptionalModel:
        return OptionalModel(x=val)

    r = f()
    assert r.x is None


def test_validation_error(tmp_cache_dir: Path) -> None:
    """Confirm that validation errors in Pydantic models propagate correctly when cached."""

    class ModelWithValidation(BaseModel):
        x: int

    @pydantic_cache
    def f(x: str) -> ModelWithValidation:
        return ModelWithValidation(x=x)  # type: ignore  # noqa: PGH003

    with pytest.raises(ValidationError):
        f("not_int")


def test_list_of_models(tmp_cache_dir: Path) -> None:
    """Ensure lists of Pydantic models are cached and compared correctly."""

    class ListModel(BaseModel):
        values: list[ResultModel]

    @pydantic_cache
    def f(n: int) -> ListModel:
        return ListModel(values=[ResultModel(value=i) for i in range(n)])

    r1 = f(3)
    r2 = f(3)
    assert r1 == r2


def test_generic_model(tmp_cache_dir: Path) -> None:
    """Check that generic Pydantic models work with caching and equality comparison."""
    T = TypeVar("T")

    class GenericModel(BaseModel, Generic[T]):
        value: T

    @pydantic_cache
    def f(x: int) -> GenericModel[int]:
        return GenericModel(value=x)

    r1 = f(10)
    r2 = f(10)
    assert r1 == r2


def test_simulated_ai_call(tmp_cache_dir: Path) -> None:
    """Simulate an AI call and verify that repeated prompts use the cache instead of recomputation."""
    calls: list[str] = []

    @pydantic_cache
    def ai_call(prompt: str) -> AIResponse:
        calls.append(prompt)
        return AIResponse(text=f"Answer: {prompt}")

    r1 = ai_call("Hello")
    r2 = ai_call("Hello")
    assert r1 == r2
    assert len(calls) == 1  # cached


def test_different_prompts(tmp_cache_dir: Path) -> None:
    """Ensure different prompt arguments generate different cached results."""

    @pydantic_cache
    def ai_call(prompt: str) -> AIResponse:
        return AIResponse(text=f"Answer: {prompt}")

    r1 = ai_call("A")
    r2 = ai_call("B")
    assert r1 != r2


def test_ai_source_change(tmp_cache_dir: Path) -> None:
    """Confirm that modifying the AI call function source creates a new cache file."""

    @pydantic_cache
    def ai_call(prompt: str) -> AIResponse:
        return AIResponse(text=f"Answer: {prompt}")

    ai_call("Hi")

    # redefine function
    @pydantic_cache
    def ai_call(prompt: str) -> AIResponse:
        return AIResponse(text=f"New Answer: {prompt}")

    ai_call("Hi")
    assert len(list(tmp_cache_dir.iterdir())) == 2


def test_corrupt_cache_file(tmp_cache_dir: Path) -> None:
    """Ensure that corrupt cache files raise a controlled validation error when loaded."""

    @pydantic_cache
    def f(x: int) -> ResultModel:
        return ResultModel(value=x)

    # write invalid JSON
    # We have to get the underlying wrapped function, otherwise we point at the wrong id.
    function_call, _ = _get_signature(f.__wrapped__, 5)
    file = tmp_cache_dir / function_call.file_name
    file.write_text("not_json")
    # Should recompute or raise controlled error
    with pytest.raises(ValidationError):
        _ = f(5)


def test_concurrent_calls(tmp_cache_dir: Path) -> None:
    """Verify that concurrent calls to the same function produce consistent results and only one cache file."""
    results: list[ResultModel] = []

    @pydantic_cache
    def f(x: int) -> ResultModel:
        return ResultModel(value=x)

    def worker() -> None:
        results.append(f(10))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r.value == 10 for r in results)
    assert len(list(tmp_cache_dir.iterdir())) == 1


def test_cache_file_exact_structure(tmp_cache_dir: Path) -> None:
    """Check that the cache file contains the exact expected structure, including arguments, name, and source hash."""

    @pydantic_cache
    def f(x: int) -> ResultModel:
        return ResultModel(value=x)

    result = f(5)
    file = next(tmp_cache_dir.iterdir())
    data = json.loads(file.read_text())

    # top-level keys
    assert set(data.keys()) == {"function_call", "result"}
    assert data["result"] == {"value": 5}
    assert result.value == 5

    fc = data["function_call"]
    assert set(fc.keys()) == {
        "function_name",
        "function_arguments",
        "function_source_code_hash",
    }

    # function_name should contain "function f" but not assert id
    assert fc["function_name"] == f.__wrapped__.__qualname__
    # enforce that no memory address digits leak into test stability

    assert fc["function_arguments"] == {"x": 5}

    # verify hash matches source code
    src = inspect.getsource(f.__wrapped__)
    src_hash = hashlib.sha256(src.encode()).hexdigest()
    assert fc["function_source_code_hash"] == src_hash


def test_cache_file_changes_when_source_changes(tmp_cache_dir: Path) -> None:
    """Ensure that cache file contents and hash change when the function source code is modified."""

    @pydantic_cache
    def f(x: int) -> ResultModel:
        return ResultModel(value=x)

    f(5)
    file1 = next(tmp_cache_dir.iterdir())
    data1 = json.loads(file1.read_text())
    hash1 = data1["function_call"]["function_source_code_hash"]

    # redefine f with different code
    @pydantic_cache
    def f(x: int) -> ResultModel:
        return ResultModel(value=x + 100)

    f(5)
    files = list(tmp_cache_dir.iterdir())
    assert len(files) == 2

    file2 = next(f for f in files if f != file1)
    data2 = json.loads(file2.read_text())
    hash2 = data2["function_call"]["function_source_code_hash"]

    # confirm different hash and updated result
    assert hash1 != hash2
    assert data2["result"] == {"value": 105}
    assert data2["function_call"]["function_arguments"] == {"x": 5}
    assert data2["function_call"]["function_name"] == f.__qualname__


def test_cache_file_valid_when_loading_same_function(tmp_cache_dir: Path) -> None:
    """Confirm that reloading the same function source does not duplicate cache files and maintains hash consistency."""

    @pydantic_cache
    def f(x: int) -> ResultModel:
        return ResultModel(value=x)

    f(5)
    file1 = next(tmp_cache_dir.iterdir())
    data1 = json.loads(file1.read_text())
    hash1 = data1["function_call"]["function_source_code_hash"]

    # redefine f with same code (like we loaded the same source code twice)
    @pydantic_cache
    def f(x: int) -> ResultModel:
        return ResultModel(value=x)

    f(5)
    files = list(tmp_cache_dir.iterdir())
    assert len(files) == 1

    file2 = next(tmp_cache_dir.iterdir())
    data2 = json.loads(file2.read_text())
    hash2 = data2["function_call"]["function_source_code_hash"]

    # confirm different hash and updated result
    assert hash1 == hash2
    assert data2["result"] == {"value": 5}
    assert data2["function_call"]["function_arguments"] == {"x": 5}
    assert data2["function_call"]["function_name"] == f.__qualname__
