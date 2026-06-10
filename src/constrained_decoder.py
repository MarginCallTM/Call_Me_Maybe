# ABOUTME: Core of constrained decoding (logit masking + token choice).
# ABOUTME: Couche 0 -> forbid tokens by setting their logit to -inf.

from collections.abc import Callable

import numpy as np

from src.tokenizer_utils import Vocabulary


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


_DIGITS = set("0123456789")


def _number_specials(vocab: Vocabulary) -> tuple[int, int]:
    """Return the IDs of the single-char '-' and '.' tokens.

    These two tokens are required to build numbers (sign and decimal
    point). If the vocabulary lacks them, the grammar cannot work, so
    we fail clearly instead of letting a None slip into a mask.

    Args:
        vocab: The loaded vocabulary
    Returns:
        A (minus_id, dot_id) tuple.
    Raises:
        DecodeError: If '-' or '.' is missing from the vocabulary.
    """
    minus_id = vocab.id_of("-")
    dot_id = vocab.id_of(".")
    if minus_id is None or dot_id is None:
        raise DecodeError(
            "Vocabulary is missing '-' or '.': cannot build numbers."
        )
    return minus_id, dot_id


def _allowed_for_number(
        state: str,
        digit_ids: set[int],
        minus_id: int,
        dot_id: int,
        stop_ids: set[int],
) -> set[int]:
    """Allowed token IDs for a number, given the current state.

    The set depends ONLY on where we are in the number grammar, never
    on the prompt. 'stop_ids' (tokens that may legally follow the number)
    are offered only in states where the number is complete.

    Args:
        state: Current state of the number state machine.
        digit_ids: IDs of all-digit tokens.
        minus_id: ID of the '-' token.
        dot_id: ID of the '.' token.
        stop_ids: IDs allowed to terminate the number.

    Returns:
        The set of token IDs allowed at this state.
    Raises:
        DecodeError: If the state is unknown(internal bug)
    """
    if state == "start":
        # A number starts with a sign or a digit (no '.' nor stop).
        return digit_ids | {minus_id}
    if state in ("int_lead", "frac_lead"):
        # Right after '-' or '.': the next token MUST be a digit.
        return set(digit_ids)
    if state == "int":
        # Valid integer so far: more digits, a '.', or we may stop.
        return digit_ids | {dot_id} | stop_ids
    if state == "frac":
        # Valid decimal so far: more digits, or we may stop.
        return digit_ids | stop_ids
    raise DecodeError(f"Unknown number state: {state!r}")


def _next_number_state(
    state: str, token_id: int, minus_id: int, dot_id: int
) -> str:
    """Compute the next state after emitting a (non-stop) token.

    Args:
        state: Current state.
        token_id: The token just chosen (never a stop token here).
        minus_id: ID of the '-' token.
        dot_id: ID of the '.' token.

    Returns:
        The next state.

    Raises:
        DecodeError: If the state is unknown (internal bug).
    """
    if state == "start":
        return "int_lead" if token_id == minus_id else "int"
    if state == "int_lead":
        return "int"
    if state == "int":
        return "frac_lead" if token_id == dot_id else "int"
    if state == "frac_lead":
        return "frac"
    if state == "frac":
        return "frac"
    raise DecodeError(f"Unknown number state: {state!r}")


def generate_number(
    get_logits: Callable[[list[int]], list[float]],
    prefix_ids: list[int],
    vocab: Vocabulary,
    stop_ids: set[int],
    max_tokens: int = 32,
) -> tuple[str, list[int]]:
    """Generate a JSON number, one constrained token at a time.

    We never trust the model to "write a number": at each step we
    compute the tokens the number grammar allows, mask everything
    else to -inf, and let the model pick the best ALLOWED token.
    The structure is guaranteed by us; the actual digits come from
    the model (i.e. from the prompt it has in its context).

    Args:
        get_logits: Maps the current input IDs to the raw next-token
        logits. We pass a function (not the model) to stay decoupled,
        testable without the heavy model, and free of any torch import in src

        prefix_ids: Token IDs already fixed before the number (prompt + JSON
        generated so far).

        vocab: The loaded vocabulary (IDs <-> token strings).
        stop_ids: Token that may legally follow the number (e.g. ',' or '}')
        Choosing one ends the number. Must be non-empty, otherwise the number
        can never terminate by model choice.

        max_tokens: Safety seatbelt against an endless loop. The full
        stop/guard
        logic is Couche 3; here it only caps length.

    Returns:
        The generated number as a string, e.g. "2" or "-3.5".

    Raises:
        DecodeError: If the grammar leaves no valid token, or the number grows
        past max_tokens.
    """
    minus_id, dot_id = _number_specials(vocab)
    # Reuse the Phase 1 predicate: non empty, all characters are
    # digits (this also accepts multi-digit chunks like "12")
    digit_ids = vocab.ids_matching(
        lambda t: t != "" and all(c in _DIGITS for c in t)
    )
    if not digit_ids:
        raise DecodeError("Vocabulary has no all-digit tokens.")

    state = "start"
    generated: list[int] = []
    for _ in range(max_tokens):
        allowed = _allowed_for_number(
            state, digit_ids, minus_id, dot_id, stop_ids
        )
        logits = get_logits(prefix_ids + generated)
        token_id = select_next_token(logits, allowed)
        if token_id in stop_ids:
            # The model chose to end the number. The stop token
            # belongs to the surrounding structure, not the number,
            # so we do NOT append it here.
            break
        generated.append(token_id)
        state = _next_number_state(
            state, token_id, minus_id, dot_id
        )
    else:
        # 'for/else': runs only if we never broke out -> runaway
        raise DecodeError("Number exceeded max_tokens (runaway)")

    # All generated tokens are digits/'-'/'.', never space-prefix
    # so we can concatenate their strings directly.
    chars = [vocab.token_for_id(tid) for tid in generated]
    number_str = "".join(c for c in chars if c is not None)
    return number_str, generated


def _is_string_content(token: str) -> bool:
    """Tell whether a token is safe inside a JSON string body.

    A token is unsafe if its text holds a double quote or a
    backslash: an unescaped quote would close the string, and
    a lone backslash would start an invalid escape. We simply
    forbid such tokens. Consequence: this Couche-1 version
    cannot produce strings that REQUIRE an escape (e.g. a quote
    inside the value); that richer case is left for later
    (the regex function).

    Args:
        token: The candidate token's text.

    Returns:
        True if the token may appear inside the string body.
    """
    return '"' not in token and "\\" not in token


def generate_string(
    get_logits: Callable[[list[int]], list[float]],
    decode: Callable[[list[int]], str],
    prefix_ids: list[int],
    vocab: Vocabulary,
    max_tokens: int = 64,
) -> tuple[str, list[int]]:
    """Generate a JSON string value, one constrained token at a time.

    A JSON string is self-delimited: it opens with a '"', holds any
    number of safe content tokens, and closes with another '"'. So,
    unlike numbers, no external stop set is needed, the closing quote
    IS the stop. We force the opening quote, allow only safe content
    (plus the closing quote) in the body, and stop on close.

    Args:
        get_logits: Maps current input IDs to raw next-token logits
            (inject model.get_logits_from_input_ids in production).
        decode: Maps content token IDs to their text (inject model.decode).
            Needed because byte-level BPE tokens are NOT plain text: spaces
            show up as 'Ġ' and bytes are remapped, so we cannot just
            concatenate them.
        prefix_ids: Token IDs already fixed before the string.
        vocab: The loaded vocabulary.
        max_tokens: Runaway seatbelt (Couche 3 owns the full guard).

    Returns:
        The decoded string value, without the surrounding quotes.

    Raises:
        DecodeError: If '"' is missing from the vocabulary, if no
        content token is allowed, or if max_tokens is exceeded
    """
    quote_id = vocab.id_of('"')
    if quote_id is None:
        raise DecodeError("Vocabulary is missing the '\"' token.")
    content_ids = vocab.ids_matching(_is_string_content)

    generated: list[int] = []  # full body, BOTH quotes included
    content: list[int] = []  # Only the tokens between the quotes
    state = "open"
    for _ in range(max_tokens):
        if state == "open":
            # Force the opening quote, nothing else
            allowed = {quote_id}
        else:
            # Inside: safe content, or the closing quote.
            allowed = content_ids | {quote_id}
        logits = get_logits(prefix_ids + generated)
        token_id = select_next_token(logits, allowed)
        generated.append(token_id)
        if state == "open":
            # We just emitted the opening quote; enter the body
            state = "body"
            continue
        if token_id == quote_id:
            # Closing quote -> the string is complete.
            break
        content.append(token_id)
    else:
        # for/else: ran only if we never broke out -> runaway.
        raise DecodeError("String exceeded max_tokens (runaway)")

    return decode(content), generated


def generate_boolean(
    get_logits: Callable[[list[int]], list[float]],
    prefix_ids: list[int],
    vocab: Vocabulary,
) -> tuple[bool, list[int]]:
    """Generate a JSON boolean in a single constrained step.

    Args:
        get_logits: Maps input IDs to raw next-token logits.
        prefix_ids: Token IDs already fixed before the bool.
        vocab: The loaded vocabulary.
    Returns:
        A (value, emitted token IDs) tuple.
    Raises:
        DecodeError: If 'true' or 'false' is missing from the vocab.
    """
    # Exactly two literals, each a single token here. We allow only
    # those two IDs; the model picks (never a keyword heuristic).
    true_id = vocab.id_of("true")
    false_id = vocab.id_of("false")
    if true_id is None or false_id is None:
        raise DecodeError(
            "Vocabulary is missing the 'true'/'false' tokens."
        )
    logits = get_logits(prefix_ids)
    token_id = select_next_token(logits, {true_id, false_id})
    return token_id == true_id, [token_id]


def generate_object(
    get_logits: Callable[[list[int]], list[float]],
    decode: Callable[[list[int]], str],
    encode_text: Callable[[str], list[int]],
    prefix_ids: list[int],
    vocab: Vocabulary,
    params: dict[str, str],
    max_value_tokens: int = 256
) -> dict[str, float | str | bool]:
    """Generate a JSON parameters object, fully constrained.

    Args:
        get_logits: Maps input IDs to raw next-token logits.
        decode: Maps token IDs to text (forwarded to generate_string).
        encode_text: Maps a fixed literal to its token IDs.
        prefix_ids: Tokens already fixed before the object.
        vocab: The loaded vocabulary.
        params: Ordered param_name -> type ("number"|"string"|
            "boolean"); order drives the JSON layout.
    Returns:
        Mapping param_name -> value (float / str / bool).
    Raises:
        DecodeError: On a missing structural token or unknown type.
    """
    # Structure (braces/keys/colons/commas) is ours: forced literals,
    # one legal token each -> no model call. Only VALUES are
    # model-driven, dispatched to the matching Couche-1 generator.
    comma_id = vocab.id_of(",")
    brace_close_id = vocab.id_of("}")
    if comma_id is None or brace_close_id is None:
        raise DecodeError("Missing ',' or '}': cannot build object.")
    number_stops = {comma_id, brace_close_id}

    result: dict[str, float | str | bool] = {}
    generated: list[int] = []
    generated += encode_text("{")
    value_tokens_used = 0
    for index, (name, ptype) in enumerate(params.items()):
        if index > 0:
            generated += encode_text(",")
        generated += encode_text(f'"{name}":')
        here = prefix_ids + generated  # context for the value

        if ptype == "number":
            number_str, value_ids = generate_number(
                get_logits, here, vocab, number_stops
            )
            result[name] = float(number_str)  # schema wants 2 -> 2.0
        elif ptype == "string":
            text_value, value_ids = generate_string(
                get_logits, decode, here, vocab
            )
            result[name] = text_value
        elif ptype == "boolean":
            bool_value, value_ids = generate_boolean(
                get_logits, here, vocab
            )
            result[name] = bool_value
        else:
            raise DecodeError(f"Unknown parameter type: {ptype!r}")

        generated += value_ids
        value_tokens_used += len(value_ids)
        if value_tokens_used > max_value_tokens:
            raise DecodeError(
                f"Object exceeded its token budget ({max_value_tokens})."
            )

    generated += encode_text("}")
    if set(result) != set(params):
        raise DecodeError("Object is missing required parameters.")
    return result
