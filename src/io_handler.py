# ABOUTME: File I/O boundary for the call-me-maybe project.
# ABOUTME: Reads/valdiates the input JSON files and writes the output

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from src.models import FunctionCall, FunctionDefinition, Prompt


class DataError(Exception):
    """Raised when data cannot be read, parsed or validated.

    This single exception type wraps every low_level failure
    (missing file, malformed JSON, schema violation) behind one
    clear message, so the entry point only needs ONE ''except DataError''
    to stay crash-free
    """


def _read_json(path: str) -> Any:
    """Read and parse a JSON file, converting failures to DataError.

    Args:
                path: Path to the JSON file to read.
    returns:
                The parsed JSON content (typically a list of dict)
    Raises:
                DataError: If the file is missing/unreadable or not valid JSON
    """
    try:
        # "with" guarantees the file is closed even or error,
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except OSError as exc:
        # Missing file, permission denied, is-a-directory, etc.
        raise DataError(f"Cannot read file '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        # Content exist but is not valid JSON (traling comma)
        raise DataError(
            f"File '{path}' is not a valid JSON: {exc}"
        ) from exc


def load_functions(path: str) -> list[FunctionDefinition]:
    """Load and validate the function definitions file.

    Args:
                path: Path to ''functions_definition.json''
    Returns:
                The list of validated FunctionDefinition objects.
    Raises:
                DataError: If the file cannot be read, parsed or validated
        against the FunctionDefinition schema.
    """
    raw = _read_json(path)
    # TypeAdapter validates the WHOLE list at once: every element must
    # match FunctionDefinition, otherwise ValidationError says which
    # element failed and why.
    adapter = TypeAdapter(list[FunctionDefinition])
    try:
        return adapter.validate_python(raw)
    except ValidationError as exc:
        raise DataError(
            f"Invalid function definitions in '{path}':\n{exc}"
        )


def load_prompts(path: str) -> list[Prompt]:
    """Load and validate the test prompt file.

    Args:
        path: Path to ''function_calling_tests.json''.
    Returns:
        The list of validated Prompts objects.
    Raises:
        DataError: If the file cannot be read, parsed
        or validated against the Prompt schema.
    """
    raw = _read_json(path)
    adapter = TypeAdapter(list[Prompt])
    try:
        return adapter.validate_python(raw)
    except ValidationError as exc:
        raise DataError(
            f"Invalid prompts in '{path}':\n{exc}"
        ) from exc


def write_results(path: str, results: list[FunctionCall]) -> None:
    """Write the function-call results to a JSON file.

    Creates the parent directory if needed and writes a valid JSON
    array (indented, no trailing comma - guaranteed by json.dump)

    Args:
        path: Destination, e.g.
            ''data/output/function_calling_results.json''
        results: The function calls to serialise.

    Raises:
        DataError: if the directory or file cannot be created/written
    """
    # model_dump() turns each pydantic object into a plain dict,
    # preserving field order: prompt, name, parameters.
    payload = [call.model_dump() for call in results]
    try:
        out_path = Path(path)
        # Create data/output (and parents) if missing no error if
        # it already exists.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            # indent=2 -> readable; ensure_ascii=False -> keep special
            # characters as is. json.dump never emits trailing commas.
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except OSError as exc:
        raise DataError(
            f"Cannot write results to '{path}': {exc}"
        ) from exc
