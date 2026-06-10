from huggingface_hub import hf_hub_download

from src.models import FunctionDefinition, ParameterSpec
from src.pipeline import process_prompt
from src.tokenizer_utils import Vocabulary


def main() -> None:
    """Drive process_prompt to fn_add_numbers, no model loaded."""
    vocab_path = hf_hub_download(
        repo_id="Qwen/Qwen3-0.6B", filename="vocab.json"
    )
    vocab = Vocabulary(vocab_path)

    # Synthetic schema: a test fixture, NOT hardcoded inside src.
    functions = [
        FunctionDefinition(
            name="fn_add_numbers",
            description="Add two numbers.",
            parameters={
                "a": ParameterSpec(type="number"),
                "b": ParameterSpec(type="number"),
            },
            returns=ParameterSpec(type="number"),
        ),
        FunctionDefinition(
            name="fn_greet",
            description="Greet someone.",
            parameters={"name": ParameterSpec(type="string")},
            returns=ParameterSpec(type="string"),
        ),
        FunctionDefinition(
            name="fn_reverse_string",
            description="Reverse a string.",
            parameters={"s": ParameterSpec(type="string")},
            returns=ParameterSpec(type="string"),
        ),
    ]

    # Char-level fake tokenizer: each character is its own token id.
    def encode_text(text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    # Real vocab ids we steer the value generation towards.
    two_id = vocab.id_of("2")
    three_id = vocab.id_of("3")
    brace_id = vocab.id_of("}")
    if two_id is None or three_id is None or brace_id is None:
        raise SystemExit("missing '2', '3' or '}' token")

    colon, a_ch, b_ch = ord(":"), ord("a"), ord("b")
    size = len(vocab)

    def get_logits(ids: list[int]) -> list[float]:
        logits = [0.0] * size
        logits[a_ch] = 50.0       # choice divergence -> fn_add_numbers
        logits[brace_id] = 100.0  # stop wins once a digit is placed
        # Peek at the forced key just before the value: it is the 4
        # chars '"' <name> '"' ':' , so the name char sits at ids[-3].
        # Each parameter then gets its OWN digit: 'b' -> 3, else -> 2.
        if len(ids) >= 3 and ids[-1] == colon and ids[-3] == b_ch:
            logits[three_id] = 50.0
        else:
            logits[two_id] = 50.0
        return logits

    call = process_prompt(
        "What is the sum of 2 and 3?",
        functions, get_logits, encode_text, decode, vocab,
    )
    print("call ->", call)
    assert call.name == "fn_add_numbers", call.name
    assert call.parameters == {"a": 2.0, "b": 3.0}, call.parameters
    assert call.prompt == "What is the sum of 2 and 3?"
    print("test_pipeline OK")


if __name__ == "__main__":
    main()
