import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

MAX_INT32 = 2147483647


allowed = Literal["string", "boolean", "number", "integer"]

class PromptItem(BaseModel):
    """One entry from function_calling_tests.json: {"prompt": "..."}"""

    # extra="forbid" replaces the old `len(prompt) == 1` check:
    # any key other than "prompt" makes validation fail.
    model_config = ConfigDict(extra="forbid")

    prompt: str

    @field_validator("prompt")
    @classmethod
    def numbers_must_fit_int32(cls, value: str) -> str:
        for num in re.findall(r"\d+", value):
            if int(num) > MAX_INT32:
                raise ValueError(
                    f"this number: {num} - is too big try small number"
                )
        return value


class Return(BaseModel):
     model_config = ConfigDict(extra="forbid")
     type: allowed


class ParamDef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: allowed


class FunctionDef(BaseModel):
    """One entry from functions_definition.json."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: Dict[str, ParamDef]
    returns: Return
    id: Optional[int] = None


class Parsing:

    @staticmethod
    def _load_json_list(path: Path, label: str) -> list:
        with open(path) as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"ERROR: {label} did not load as a list")
            sys.exit(1)

        if len(data) == 0:
            print(f"ERROR: {label} file is empty")
            sys.exit(1)

        return data

    @staticmethod
    def valid_prompt(path) -> List[PromptItem]:
        raw = Parsing._load_json_list(
            Path(path),
            "function_calling_tests.json",
        )

        prompts: List[PromptItem] = []
        for item in raw:
            try:
                prompts.append(PromptItem.model_validate(item))
            except ValidationError as e:
                print(f"ERROR: {item} is invalid -> {e}")
                sys.exit(1)

        return prompts

    @staticmethod
    def valid_function_def(path) -> List[FunctionDef]:
        raw = Parsing._load_json_list(Path(path), "functions_definition.json")

        functions: List[FunctionDef] = []
        for item in raw:
            try:
                functions.append(FunctionDef.model_validate(item))
            except ValidationError as e:
                print(f"ERROR: {item} is invalid -> {e}")
                sys.exit(1)

        return functions

    def set_id(self) -> List[FunctionDef]:
        path = Path("data/input/functions_definition.json")
        functions = self.valid_function_def(path)

        for i, func in enumerate(functions):
            func.id = i

        return functions
