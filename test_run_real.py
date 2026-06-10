from src.io_handler import load_functions, load_prompts
from src.pipeline import run


def main() -> None:
    """Smoke test: run() avec le vrai modele, sur 2 prompts."""
    functions = load_functions("data/input/functions_definition.json")
    prompts = load_prompts("data/input/function_calling_tests.json")

    results = run(functions, prompts[:2])

    for call in results:
        print(call)


if __name__ == "__main__":
    main()
