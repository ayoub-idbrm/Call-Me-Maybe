import json
from pathlib import Path
from typing import Any

# prompt = Path("aidbrm/ayoub/call/data/input/function_calling_tests.json").absolute()
# func = Path("data/functions_definition.json").absolute()


def load_prompt():
    # with prompt.open('r', encoding='utf-8') as f:
    #     return json.load(f)
    with open ("data/input/function_calling_tests.json", "r") as f:
        data = json.load(f)
        return data


def load_funct():
    with open ("data/input/functions_definition.json", "r") as d:
        data = json.load(d)
        return data


def main():
    test = load_prompt()
    print(test)
    test2 = load_funct()
    print(test2)

if __name__ == "__main__":
    main()