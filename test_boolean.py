from collections.abc import Callable

from huggingface_hub import hf_hub_download

from src.constrained_decoder import generate_boolean
from src.tokenizer_utils import Vocabulary


def constant_logits(
    favourite: int, size: int
) -> Callable[[list[int]], list[float]]:
    """Fake get_logits giving the top score to one token id."""
    def get_logits(input_ids: list[int]) -> list[float]:
        logits = [0.0] * size
        logits[favourite] = 100.0
        return logits

    return get_logits


def main() -> None:
    """Check the boolean is forced to 'true'/'false' only."""
    vocab_path = hf_hub_download(
        repo_id="Qwen/Qwen3-0.6B", filename="vocab.json"
    )
    vocab = Vocabulary(vocab_path)

    prefix = [1, 2, 3]
    size = len(vocab)

    # 1) Model prefers 'false' -> we must get False.
    false_id = vocab.id_of("false")
    if false_id is None:
        raise SystemExit("missing 'false' token")
    fav_false = constant_logits(false_id, size)
    print("prefers false ->", generate_boolean(fav_false, prefix, vocab))

    # 2) Model prefers a digit. It is FORBIDDEN, so the choice falls
    #    back to the best ALLOWED among {true, false}. Both tie at
    #    0.0, so argmax takes the smaller id -> 'true' (1866 < 3849).
    digit_id = vocab.id_of("5")
    if digit_id is None:
        raise SystemExit("missing '5' token")
    fav_digit = constant_logits(digit_id, size)
    print("prefers '5'   ->", generate_boolean(fav_digit, prefix, vocab))


if __name__ == "__main__":
    main()
