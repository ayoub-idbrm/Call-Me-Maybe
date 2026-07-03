import json
from pathlib import Path
from typing import Any

prompt = Path("data/function_calling_tests.json").absolute()
func = Path("data/functions_definition.json").absolute()

def load_prompt():
    with prompt.open('r', encoding='utf-8') as f:
        return json.load(f)

def main():
    test = load_prompt()
    print(test)

main()