from collections.abc import Callable

from huggingface_hub import hf_hub_download

from src.constrained_decoder import generate_string
from src.tokenizer_utils import Vocabulary


def make_fake_logits(
        prefix_len: int, script: list[int], size: int
) -> Callable[[list[int]], list[float]]:
    """Fake get_logits favouring a scripted token sequence."""
    def get_logits(input_ids: list[int]) -> list[float]:
        step = len(input_ids) - prefix_len
        logits = [0.0] * size
        if 0 <= step < len(script):
            logits[script[step]] = 100.0
        return logits

    return get_logits


def main() -> None:
    """Generate the string "Bob" from a scripted fake model."""
    vocab_path = hf_hub_download(
        repo_id="Qwen/Qwen3-0.6B", filename="vocab.json"
    )
    vocab = Vocabulary(vocab_path)

    # Fake decoder: join token texts and turn the space marker into
    # a real space. Good enough to test WITHOUT the heavy model.
    def fake_decode(ids: list[int]) -> str:
        parts = [vocab.token_for_id(i) or "" for i in ids]
        return "".join(parts).replace("Ġ", " ")

    # Full token sequence the fake model "wants": " Bob "
    script: list[int] = []
    for ch in ['"', "B", '"', "o", "o" "b", "s", '"']:
        tid = vocab.id_of(ch)
        if tid is None:
            raise SystemExit(f"missing token {ch!r}")
        script.append(tid)

    prefix = [1, 2, 3]
    get_logits = make_fake_logits(len(prefix), script, len(vocab))

    value = generate_string(get_logits, fake_decode, prefix, vocab)
    print("generated string ->", repr(value))  # expect 'Bob'


if __name__ == "__main__":
    main()

