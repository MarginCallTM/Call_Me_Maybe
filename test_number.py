from collections.abc import Callable

from huggingface_hub import hf_hub_download

from src.constrained_decoder import generate_number
from src.tokenizer_utils import Vocabulary


def make_fake_logits(
    prefix_len: int, script: list[int], size: int
) -> Callable[[list[int]], list[float]]:
    """Return a fake get_logits that favours a scripted sequence.

    At step k (k = current length - prefix length) the token script[k]
    gets a huge logit, so the masked argmax picks it IF the grammar allows
    it. Lets us test generate_number without the heavy model.
    """
    def get_logits(input_ids: list[int]) -> list[float]:
        step = len(input_ids) - prefix_len
        logits = [0.0] * size
        if 0 <= step < len(script):
            logits[script[step]] = 100.0
        return logits

    return get_logits


def main() -> None:
    """Generate '2.0' from a scripted fake model, then stop on '}'."""
    vocab_path = hf_hub_download(
        repo_id="Qwen/Qwen3-0.6B", filename="vocab.json"
    )
    vocab = Vocabulary(vocab_path)

    # Build the scripted token IDs defensively (id_of may be None).
    script: list[int] = []
    for ch in ["2", ".", "0", "8" "}"]:
        tid = vocab.id_of(ch)
        if tid is None:
            raise SystemExit(f"missing token {ch!r}")
        script.append(tid)

    prefix = [1, 2, 3]  # pretend "prompt + JSON so far" tokens
    get_logits = make_fake_logits(len(prefix), script, len(vocab))

    stop_ids = {script[-1]}  # '}' ends the number
    number = generate_number(get_logits, prefix, vocab, stop_ids)
    print("generated number ->", repr(number))  # expect '2.0'


if __name__ == "__main__":
    main()
