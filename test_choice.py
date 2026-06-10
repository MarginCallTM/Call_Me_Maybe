from collections.abc import Callable

from src.constrained_decoder import generate_choice


def make_fake_logits(
    prefix_len: int, target: str, size: int
) -> Callable[[list[int]], list[float]]:
    """Fake get_logits steering generation toward target's chars."""
    def get_logits(input_ids: list[int]) -> list[float]:
        step = len(input_ids) - prefix_len
        logits = [0.0] * size
        if 0 <= step < len(target):
            logits[ord(target[step])] = 100.0
        return logits

    return get_logits


def main() -> None:
    """Pick 'fn_greet' from the function names, no model involved."""
    def fake_encode_text(text: str) -> list[int]:
        return [ord(ch) for ch in text]  # each char is its own id

    choices = [
        "fn_add_numbers",
        "fn_greet",
        "fn_reverse_string",
        "fn_get_square_root",
        "fn_substitute_string_with_regex",
    ]
    prefix = [1, 2, 3]
    target = "fn_greet"
    get_logits = make_fake_logits(len(prefix), target, 256)

    chosen = generate_choice(
        get_logits, fake_encode_text, prefix, choices
    )
    print("chosen ->", chosen)
    assert chosen == "fn_greet", chosen
    print("test_choice OK")


if __name__ == "__main__":
    main()
