from huggingface_hub import hf_hub_download
from src.tokenizer_utils import SPACE_MARKER, Vocabulary


def main() -> None:
    """Build a Vocabulary and check a few know token IDs"""
    vocab_path = hf_hub_download(
        repo_id="Qwen/Qwen3-0.6B", filename="vocab.json"
    )
    vocab = Vocabulary(vocab_path)

    print("vocab size: ", len(vocab))  # expect 151643

    # Structural singel-char tokens (from Phase 1)
    for ch in ["{", "}", "\"", ":", ","]:
        print(f"id_of({ch!r}) =", vocab.id_of(ch))

    # Reverse lookup + a special/padding ID
    print("token_for_id(90) =", repr(vocab.token_for_id(90)))
    print("token_for_id(151900) =", vocab.token_for_id(151900))

    # The constrained-decoding bridge: token made ONLY of digits.
    digits = set("0123456789")
    digit_ids = vocab.ids_matching(
        lambda t: t != "" and all(c in digits for c in t)
    )
    print("all-digit tokens   :", len(digit_ids))
    print("'5' (id 20) in set :", vocab.id_of("5") in digit_ids)

    # The space marker (G) demo.
    print("is_space_prefixed('sum') ->",
          vocab.is_space_prefixed("sum"))
    print("is_space_prefixed(G+'sum') ->",
          vocab.is_space_prefixed(SPACE_MARKER + "sum"))


if __name__ == "__main__":
    main()
