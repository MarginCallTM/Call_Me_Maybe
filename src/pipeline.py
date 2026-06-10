# ABOUTME: Pipeline: turn each prompt into a structured FunctionCall.
# ABOUTME: Wires the real model into the constrained decoders.

import sys
from collections.abc import Callable

from llm_sdk.llm_sdk import Small_LLM_Model

from src.constrained_decoder import (
    DecodeError,
    generate_choice,
    generate_object,
)
from src.models import FunctionCall, FunctionDefinition, Prompt
from src.tokenizer_utils import Vocabulary


def _functions_catalog(functions: list[FunctionDefinition]) -> str:
    """Return a 'name: description' line per function """
    return "\n".join(f"{f.name}: {f.description}" for f in functions)


def build_choice_prompt(
        functions: list[FunctionDefinition], user_prompt: str
) -> str:
    """Build the prompt that elicits the function NAME."""
    return (
        "You are a function-calling engine. Choose the single best "
        "function for the user request.\n"
        "Available functions:\n"
        f"{_functions_catalog(functions)}\n"
        f"User request: {user_prompt}\n"
        "Function name: "
    )


def build_params_prompt(
        functions: list[FunctionDefinition],
        user_prompt: str,
        name: str,
) -> str:
    """Build the prompt that primes the arguments JSON object."""
    return (
        "You are a function-calling engine. Extract the arguments "
        "for the chosen function from the user request.\n"
        "Available functions:\n"
        f"{_functions_catalog(functions)}\n"
        f"User request: {user_prompt}\n"
        f"Function name: {name}\n"
        "Argument JSON: "
    )


def process_prompt(
        user_prompt: str,
        functions: list[FunctionDefinition],
        get_logits: Callable[[list[int]], list[float]],
        encode_text: Callable[[str], list[int]],
        decode: Callable[[list[int]], str],
        vocab: Vocabulary,
) -> FunctionCall:
    """Turn one prompt into a FunctionCall (choice then arguments.)"""
    names = [f.name for f in functions]
    if not names:
        raise DecodeError("No functions available to choose from.")

    # 1) The MODEL chooses the function (constrained to real names).
    choice_text = build_choice_prompt(functions, user_prompt)
    name = generate_choice(
        get_logits, encode_text, encode_text(choice_text), names
    )
    chosen = next(f for f in functions if f.name == name)

    # 2) Generate This function's arguments under its schema.
    params_text = build_params_prompt(functions, user_prompt, name)
    schema: dict[str, str] = {
        pname: spec.type for pname, spec in chosen.parameters.items()
    }
    values = generate_object(
        get_logits, decode, encode_text,
        encode_text(params_text), vocab, schema,
    )
    return FunctionCall(
        prompt=user_prompt, name=name, parameters=values
    )


def run(
    functions: list[FunctionDefinition], prompts: list[Prompt]
) -> list[FunctionCall]:
    """Process every prompt, skipping (with a warning) any failure"""
    # Heavy: load the Qwen weights (first run also DL them)
    # Everything stays torch-free in src thanks to typed wrappers.
    model = Small_LLM_Model()
    vocab = Vocabulary(model.get_path_to_vocab_file())

    def get_logits(ids: list[int]) -> list[float]:
        logits: list[float] = model.get_logits_from_input_ids(ids)
        return logits

    def encode_text(text: str) -> list[int]:
        ids: list[int] = model.encode(text).tolist()[0]
        return ids

    def decode(ids: list[int]) -> str:
        text: str = model.decode(ids)
        return text

    results: list[FunctionCall] = []
    for item in prompts:
        try:
            results.append(
                process_prompt(
                    item.prompt, functions,
                    get_logits, encode_text, decode, vocab
                )
            )
        except DecodeError as exc:
            print(
                f"[warn] skipped {item.prompt!r}: {exc}",
                file=sys.stderr,
            )
    return results
