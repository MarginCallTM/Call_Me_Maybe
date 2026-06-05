# ABOUTME: Throwaway discovery script to SEE how the LLM works step by step.
# ABOUTME: Not part of the src/, not submitted. Run: uv run python explore.py

# We import ONLY the public SDK wrapper and the standard json module.
# Note: we deliberately do NOT "import torch" -- even here -- to mirror the
# project rule. encode() returns a torch tensor, but we can call .tolist()
# on the object we receive without importing torch ourselves.

import json

from llm_sdk import Small_LLM_Model


def main() -> None:
    # STEP 0: Load the model (first run DL Qwen)
    print("Loading model (first run DL the weight)...")
    model = Small_LLM_Model()

    # STEP 1: A prompt in plain text (What a humain writes)
    prompt = "What is the sum of 2 and 3?"
    print("\n[1] Prompt:", repr(prompt))

    # STEP 2: Tokenization -> Input IDs
    # encode() returns a 2-D tensor of shape [1, n]. We turn it
    # into a plain Python list[int] with .tolist()[0] (no torch)
    ids_tensor = model.encode(prompt)
    input_ids = ids_tensor.tolist()[0]
    print("\n[2] Input IDs (the model only sees these numbers):")
    print("    ", input_ids)
    print("    -> the prompt became", len(input_ids), "tokens")

    # STEP 3: Logits: one score per token in the whole vocabulary
    # get_logits_from_input_ids takes a list[int] and returns list[float]
    logits = model.get_logits_from_input_ids(input_ids)
    print("\n[3] logits: one score per possible next token.")
    print("    vocabulary size =", len(logits), "scores")

    # STEP 4: Greedy decoding: the next token is the index of the max
    # logit. This is just "find the index of the biggest value", like in C
    next_id = max(range(len(logits)), key=lambda i: logits[i])
    print("\n[4] argmax (greedy): highest-scoring token id= ", next_id)
    print("    its logit value =", logits[next_id])

    # STEP 5: Translate that id back to text, two different ways:
    # (a) the SDK decoder, (b) the raw vocab.json mapping.
    # vocab.json maps "token-string" -> id, so we invert it to id
    # -> string. Watch the 'G with a dot' symbol that marks a leading space
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)  # dict: token_string -> id
    id_to_token = {i: t for t, i in vocab.items()}

    print("\n[5] The predicted next token:")
    print("     via decode(): ", repr(model.decode([next_id])))
    print("     via vocab.json: ", repr(id_to_token.get(next_id)))

    # Bonus: the model's TOP 5 candidates -> show it ranks all tokens.
    top5 = sorted(range(len(logits)), key=lambda i: logits[i],
                  reverse=True)[:5]
    print("\n[bonus] Top 5 candidate next tokens (id | token | logit):")
    for i in top5:
        print("    ", i, repr(id_to_token.get(i)), round(logits[i], 3))

    experiment_masking(model, input_ids, logits, id_to_token)


def experiment_masking(
        model: Small_LLM_Model,
        inputs_ids: list[int],
        logits: list[float],
        id_to_token: dict[int, str],
) -> None:
    """Show constrained decoding in action by allowing only digit tokens"""

    # STEP 1: Find wich IDs in the vocabulary correspond to '0'..'9'
    # The tokens for bare digits have NO leading G (they appear after
    # the space token 220). We look for strings that are exactly one
    # of the ten digit characters.
    digit_strings = set("0123456789")
    allowed_ids = [
        i for i, t in id_to_token.items()
        if t in digit_strings
    ]
    print("\n--- Masking experiment: force a digit ---")
    print("Allowed token IDs (digits 0-9)",
          [(i, id_to_token[i]) for i in sorted(allowed_ids)])

    # STEP 2: Copy logits (never mutate the original list)
    masked = list(logits)

    # STEP 3: Set every non-allowed token to -inf
    allowed_set = set(allowed_ids)
    for i in range(len(masked)):
        if i not in allowed_set:
            masked[i] = float('-inf')

        # STEP 4: argmax among what remains
    best_id = max(allowed_ids, key=lambda i: masked[i])
    print("Forced next token: ", repr(id_to_token[best_id]),
          "| logit:", round(logits[best_id], 3))
    print("(without masking the model would have picked: ",
          repr(id_to_token.get(
              max(range(len(logits)), key=lambda i: logits[i])
          )), ")")


if __name__ == "__main__":
    main()
