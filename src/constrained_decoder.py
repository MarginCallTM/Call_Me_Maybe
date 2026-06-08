# ABOUTME: Core of constrained decoding (logit masking + token choice).
# ABOUTME: Couche 0 -> forbid tokens by setting their logit to -inf.

import numpy as np


class DecodeError(Exception):
    """Raised when the decoder reaches an impossible state.

    Mirrors DataError but for generation logic (e.g. no token is
    allowed at a position). Lets the entry point stay crash-free.
    """


def mask_logits(
    logits: list[float], allowed_ids: set[int]
) -> np.ndarray:
    """Return a copy of *logits* where forbidden tokens are -inf.

    We start from an all -inf array (EVERY token forbidden by default),
    then "open" only the allowed positions by copying their real
    logit back. A forbidden token can therefore never be chosen,
    because -inf is always the lowest possible score.

    Args:
        logits: Raw next-token scores from the model (length = full
            vocabulary, e.g. 151936).
        allowed_ids: The token IDs permitted at this position.

    Returns:
        A numpy array of the same length, with allowed logits kept and
        all others set to negative infinity.
    """
    # Everything starts forbidden.
    masked = np.full(len(logits), -np.inf, dtype=np.float64)
    if not allowed_ids:
        return masked
    # Vectorised: bring back the real scores ONLY for allowed IDs.
    index = np.fromiter(allowed_ids, dtype=np.int64)
    source = np.array(logits, dtype=np.float64)
    masked[index] = source[index]
    return masked


def select_next_token(
    logits: list[float], allowed_ids: set[int]
) -> int:
    """Pick the highest-scoring token AMONG the allowed ones.

    Greedy decoding: after masking, every forbidden token is -inf, so
    argmax necessarily lands on an allowed token.

    Args:
        logits: Raw next-token scores from the model.
        allowed_ids: The token IDs permitted at this position.

    Returns:
        The chosen token ID.

    Raises:
        DecodeError: If no token is allowed (grammar/decoder bug).
    """
    if not allowed_ids:
        raise DecodeError(
            "No allowed token at this position: the grammar left "
            "no valid choice (decoder bug)."
        )
    masked = mask_logits(logits, allowed_ids)
    # argmax = index of the max score = the best ALLOWED token,
    # since all the others are -inf.
    return int(np.argmax(masked))