# ABOUTME: Pydantic data models for the call-me-maybe project.
# ABOUTME: They validate the input files and the structued output

from typing import Literal
from pydantic import BaseModel, ConfigDict

# The set of parameter types we know how to handle. Using a
# literal means pydantic REJECTS any unknow type with a clear
# ValidationError, wich we will catch gracefully when loading output file.
ParameterType = Literal["number", "string", "boolean"]


class ParameterSpec(BaseModel):
    """Type descriptor of a single parameter (the schema side)

    In "functions_definition.json" every parameter maps to an
    object such as "{"type": "number"}". This model validates that the
    "type" field is present and is one of the allowed values.

    Attributes:

    type:
            The expected type of the parameter ("number", "string" or "bool")
    """
    # extra="forbid": reject any unexpected key (e.g typo) instead
    # of silently ignoring it -> fail fast with a clear error.
    model_config = ConfigDict(extra="forbid")

    type: ParameterType


class FunctionDefinition(BaseModel):
    """One callable function as described in function_definition.json.

    Attributes:
    name:
            Unique function name, e.g. "fn_add_numbers".
    description:
            Human-readable description of what the function does.
    parameters:
            Mapping "param_name" -> ParameterSpec. Each value describes
            the EXPECTED type of that parameter (the schema, not a value).
    returns:
            Type descriptor of the return value. Modelled for completeness
            and validation; the pipeline itself does not use it.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec


class Prompt(BaseModel):
    """A single natural-language request from the test file.

    Attributes
    prompt:
                The raw question, e.g. "What is the sum of 2 and 3?"
    """

    model_config = ConfigDict(extra="forbid")
    prompt: str


class FunctionCall(BaseModel):
    """The structured result produced for one prompt (the output side).

    Note the difference with :class:`FunctionDefinition`: here
    ``parameters`` holds the ACTUAL extracted values, not type
    descriptors.

    Attributes
    prompt:
        The original natural-language request (echoed back).
    name:
        The chosen function name.
    parameters:
        Mapping ``param_name -> value``. A value is a float (number),
        a str (string) or a bool (boolean).
    """

    model_config = ConfigDict(extra="forbid")
    
    prompt: str
    name: str
    parameters: dict[str, float | str | bool]
