# ABOUTME: Vocabulary helper mapping token strings <-> integer IDs.
# ABOUTME: Foundation for constrained decoding (which IDs are allowed).

import json
from collections.abc import Callable

from src.io_handler import DataError

# Byte-level BPE marker (Unicode U+0120, shown as a crossed "G"):
# the tokenizer puts it before a word to mean "a space precedes it",
# so a bare word and its space-prefixed form are TWO different tokens.
SPACE_MARKER = "Ġ"


class Vocabulary:
    """Two-way mapping between token strings and integer IDs.

    The model only ever works with integer IDs; this class translates
    back and forth and answers the core question of constrained
    decoding: "which token IDs are allowed at this position?".

    Attributes:
        token_to_id: Mapping token string -> integer ID (native order
            of vocab.json).
        id_to_token: Reverse mapping integer ID -> token string.
    """

    def __init__(self, vocab_path: str) -> None:
        """Load vocab.json and build both lookup tables.

        Args:
            vocab_path: Path to the tokenizer vocab.json file (the
                value returned by the SDK's get_path_to_vocab_file()).

        Raises:
            DataError: If the file is missing, unreadable or invalid.
        """
        self.token_to_id: dict[str, int] = self._load(vocab_path)
        # Build the reverse table once -> later id->token is O(1).
        self.id_to_token: dict[int, str] = {
            tid: tok for tok, tid in self.token_to_id.items()
        }

    @staticmethod
    def _load(vocab_path: str) -> dict[str, int]:
        """Read vocab.json into a token-string -> ID dictionary.

        Args:
            vocab_path: Path to vocab.json.

        Returns:
            The token-string -> integer-ID mapping.

        Raises:
            DataError: If the file cannot be read or parsed.
        """
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                data: dict[str, int] = json.load(f)
                return data
        except OSError as exc:
            raise DataError(
                f"Cannot read vocab file '{vocab_path}': {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise DataError(
                f"Vocab file '{vocab_path}' is not valid JSON: {exc}"
            ) from exc

    def __len__(self) -> int:
        """Return the number of known tokens (vocab.json entries)."""
        return len(self.token_to_id)

    def token_for_id(self, token_id: int) -> str | None:
        """Return the token string for an ID, or None if unknown.

        The logits array is larger than vocab.json: the extra IDs are
        special/padding tokens with no text. They return None here and
        MUST be masked out (-inf) by the decoder.

        Args:
            token_id: The integer ID to look up.

        Returns:
            The token string, or None if the ID has no text.
        """
        return self.id_to_token.get(token_id)

    def id_of(self, token: str) -> int | None:
        """Return the ID of an exact token string, or None if absent.

        Useful for structural characters whose token is a single char,
        e.g. id_of("{") -> 90.

        Args:
            token: The exact token string to look up.

        Returns:
            The integer ID, or None if the token does not exist.
        """

        return self.token_to_id.get(token)

    def ids_matching(
        self, predicate: Callable[[str], bool]
    ) -> set[int]:
        """Return every token ID whose text satisfies *predicate*.

        This is the bridge to constrained decoding: the decoder passes
        a rule (e.g. "every character is a digit") and gets back the
        set of allowed token IDs for the current position. A token may
        span several characters, so multi-char tokens like "12" are
        included when they satisfy the rule.

        Args:
            predicate: Function token_string -> bool. True means the
                token is allowed at the current position.

        Returns:
            The set of allowed token IDs.
        """
        return {
            tid
            for tok, tid in self.token_to_id.items()
            if predicate(tok)
        }

    def is_space_prefixed(self, token: str) -> bool:
        """Tell whether a token string starts with the space marker.

        Byte-level BPE encodes a leading space as SPACE_MARKER (G),
        not as a real space. This helper makes that explicit so later
        phases never confuse "sum" with its space-prefixed variant.

        Args:
            token: The token string to test.

        Returns:
            True if the token represents a space-prefixed word.
        """
        return token.startswith(SPACE_MARKER)
