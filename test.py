def main() -> None:
    """Show that 'with open' always closes the file, even on error."""
    path = "data/input/functions_definition.json"

    # 1) Normal case
    # We read inside the block, then check f.closed before/after.
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        print("Inside block -> f.closed =", f.closed)
    print("After block -> f.closed =", f.closed)
    print("Read", len(content), "characters from the file")
    print("-" * 50)

    # 2) Error case
    # We raise an error INSIDE the block. The context manager must
    # still close the file on the way out. We keep a reference to the
    # file object (f_ref) so we can inspect it AFTER the crash
    f_ref = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            f_ref = f
            raise ValueError("boom: an error happened inside 'with'")
    except ValueError as exc:
        print("Caught the error:", exc)
    print("After error -> f_ref.closed =", f_ref.closed)


if __name__ == "__main__":
    main()
