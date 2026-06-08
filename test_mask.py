from src.constrained_decoder import (
    DecodeError,
    mask_logits,
    select_next_token,
)


def main() -> None:
    """Check masking + greedy choice on tiny fake logits."""
    logits = [1.0, 5.0, 3.0, 9.0, 2.0, 4.0]

    # Case 1: allow {0, 2, 5}. Best ALLOWED is id 5 (4.0)
    # NOT id 3 (9.0) which is forbidden
    allowed = {0, 2, 5}
    print("masked : ", mask_logits(logits, allowed))
    print("chosen token: ", select_next_token(logits, allowed))

    # Case 2: a single allowed id -> must pick it
    print("single allow:", select_next_token(logits, {3}))

    # Case 3: empty allowed set -> clean DecodeError, no crash.
    try:
        select_next_token(logits, set())
    except DecodeError as exc:
        print("DecodeError :", exc)


if __name__ == "__main__":
    main()
