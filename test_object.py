from collections.abc import Callable

from huggingface_hub import hf_hub_download

from src.constrained_decoder import generate_object
from src.tokenizer_utils import Vocabulary


def scripted_logits(
    favourites: list[int], size: int
) -> Callable[[list[int]], list[float]]:
    """Fake get_logits favouring favourites[k] on the k-th call."""
    state = {"k": 0}

    def get_logits(input_ids: list[int]) -> list[float]:
        logits = [0.0] * size
        k = state["k"]
        if k < len(favourites):
            logits[favourites[k]] = 100.0
        state["k"] += 1
        return logits

    return get_logits


def main() -> None:
    """Build {"a": 2.0, "b": 3.0} from a scripted fake model."""
    vocab_path = hf_hub_download(
        repo_id="Qwen/Qwen3-0.6B", filename="vocab.json"
    )
    vocab = Vocabulary(vocab_path)

    # Resolve the token ids we will steer the fake model towards.
    ids: dict[str, int] = {}
    for ch in ["2", "3", "}"]:
        tid = vocab.id_of(ch)
        if tid is None:
            raise SystemExit(f"missing token {ch!r}")
        ids[ch] = tid

    # Per call: '2' then a stop, '3' then a stop ('}' acts as stop).
    favourites = [ids["2"], ids["}"], ids["3"], ids["}"]]
    get_logits = scripted_logits(favourites, len(vocab))

    # Record the forced literals to check the structure is ours.
    forced: list[str] = []

    def fake_encode_text(text: str) -> list[int]:
        forced.append(text)
        return []

    def fake_decode(token_ids: list[int]) -> str:
        return ""  # never used: no string param in this schema

    params = {"a": "number", "b": "number"}
    result = generate_object(
        get_logits,
        fake_decode,
        fake_encode_text,
        prefix_ids=[1, 2, 3],
        vocab=vocab,
        params=params,
    )

    print("result ->", result)
    print("forced ->", forced)
    assert result == {"a": 2.0, "b": 3.0}, result
    assert forced == ["{", '"a":', ",", '"b":', "}"], forced
    print("test_object OK")


if __name__ == "__main__":
    main()
