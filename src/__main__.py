try:
    from llm_sdk.llm_sdk import Small_LLM_Model
    from pathlib import Path
    from typing import Any, Tuple
    from pydantic import BaseModel, ConfigDict
    from src.parsing import Parsing, PromptItem, FunctionDef
    from numpy import argmax
    from math import inf
    import json
    import re
    import sys
    import time
except BaseException as i:
    print(f"ERROR: {i}")


class LLM(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Small_LLM_Model = Small_LLM_Model()

    def set_param(self, function_def: FunctionDef) -> list[Tuple[str, Any]]:
        return list(function_def.parameters.items())

    def extract_num_in_txt(self, text: object) -> list[str]:
        if isinstance(text, PromptItem):
            text = text.prompt
        elif isinstance(text, dict):
            text = text.get("prompt", "")
        return re.findall(r'-?\d+(?:\.\d+)?', str(text))

    def cons_dec(self, arr: list[float]) -> list[float]:
        numb = set()
        for ch in "-0123456789":
            ids = self.model.encode(ch).squeeze().tolist()
            if isinstance(ids, int):
                numb.add(ids)
            else:
                numb.update(ids)
        for i in range(len(arr)):
            if i not in numb:
                arr[i] = -inf
        return arr

    def cons_param_str(self, arr: list[float]) -> list[float]:
        allowed_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " .,'\"!?-_*+^$[]{}()|\\/:;=<>%&#@~`"
        )

        for i in range(len(arr)):
            text = self.model.decode(i)
            if not text or any(c not in allowed_chars for c in text):
                arr[i] = -inf
        return arr

    def cons_param(self, arr: list[float]) -> list[float]:
        allowed_chars = set("-0123456789.,")
        for i in range(len(arr)):
            text = self.model.decode(i)
            if not text or any(c not in allowed_chars for c in text):
                arr[i] = -inf
        return arr

    def extractparam(
        self, prompt: str, param_type: str = "number"
    ) -> list[int]:
        buff = self.model.encode(prompt).squeeze().tolist()
        leen = len(buff)

        for _ in range(20):
            logits = self.model.get_logits_from_input_ids(buff)

            if param_type == "string":
                logits = self.cons_param_str(logits)
            else:
                logits = self.cons_param(logits)

            token = int(argmax(logits))
            text = self.model.decode(token)

            if "," in text or "\n" in text or "\\n" in text:
                break

            buff.append(token)

        return list(buff[leen:])

    def processing(
        self, prompt_path: Path, func_path: Path
    ) -> list[dict[str, Any]]:
        prompt_items = Parsing.valid_prompt(prompt_path)
        pars = Parsing()
        data = pars.set_id()

        results = []

        function_list = "\n".join(
            f"{i}: {d.name} - takes parameters: {list(d.parameters.keys())}"
            for i, d in enumerate(data)
        )

        general_prompt = (
            "You are a function selector. Read \
                the user text and pick the number "
            "of the function that matches it.\n\n"
            "Here are the available functions:\n"
            f"{function_list}\n\n"
            "Rules:\n"
            "- If no function matches, or the number \
                of parameters or their type "
            "does not match, output -7.\n\n"
            "Examples:\n"
            "user txt = what is the sum of 33 and 4660? output=0\n"
            "user txt = what is the square root of 1662? output=3\n"
            "user txt = Reverse the string 'world' output=2\n"
            "user txt = Greet shrek output=1\n"
            "user txt = what is the sum of 0? output=-7\n"
            "user txt = what is the sum of a? output=-7\n\n"
            "user txt = "
        )
        enc_gen = self.model.encode(general_prompt).squeeze().tolist()

        for index, prompt_item in enumerate(prompt_items):
            print(f"processing: {index}/11")
            prompt = prompt_item.prompt

            buffer = list(enc_gen)
            buffer.extend(self.model.encode(f"{prompt} \
                output=").squeeze().tolist())

            logits = self.model.get_logits_from_input_ids(buffer)
            logits = self.cons_dec(logits)
            token_id = int(argmax(logits))
            buffer.append(token_id)

            try:
                func_id = int(self.model.decode(token_id))
            except BaseException:
                print("function not found")
                sys.exit(1)

            if func_id < 0 or func_id >= len(data):
                print("no matching function found")
                continue

            parameter = self.set_param(data[func_id])
            is_str_func = (
                parameter[0][1].type == "string" if parameter else False)

            if not is_str_func:
                numbers = self.extract_num_in_txt(prompt)
                params_dict_num = {}
                for idx, (p_name, p_info) in enumerate(parameter):
                    num_val: float | None = float(numbers[idx]) \
                        if idx < len(numbers) else None
                    params_dict_num[p_name] = num_val
                results.append({
                    "prompt": prompt,
                    "name": data[func_id].name,
                    "parameters": params_dict_num
                })
                continue

            param_prompt = f"""You are a parameter extraction model.

                    Your only task is to extract the \
                        parameters for the selected function.

                    Selected function:
                    {data[func_id].name}

                    Required parameters:
                    {data[func_id].parameters}

                    Rules:
                    1. Extract ONLY the required parameters.
                    2. Do NOT answer the user's question.
                    3. Do NOT explain anything.
                    4. Do NOT invent missing values.
                    5. Keep the original value exactly \
                        as written whenever possible.
                    6. Strings must be strings without changing their content.
                    7. Output the parameters in the same \
                        order as the function definition.
                    8. If a required parameter cannot be found, output MISSING.
                    9. If there are multiple values of the \
                        same type, choose the ones that \
                            belong to the user's request.
                    10. Output each parameter as parameter_name:value.
                    11. Separate consecutive parameters with ",\n".
                    12. Do not output anything except the extracted parameters.
                    13. Strip surrounding quotes (single or double) \
                        from string values, keep the inner content \
                            exactly as written.

                    Examples

                    Function:
                    fn_greet

                    Parameters:
                    name:string

                    User:
                    Greet John

                    Output:
                    name:John

                    Function:
                    fn_greet

                    Parameters:
                    name:string

                    User:
                    Greet shrek

                    Output:
                    name:shrek

                    Function:
                    fn_reverse_string

                    Parameters:
                    s:string

                    User:
                    Reverse the string "hello"

                    Output:
                    s:hello

                    Function:
                    fn_reverse_string

                    Parameters:
                    s:string

                    User:
                    Reverse the string 'world'

                    Output:
                    s:world

                    Function:
                    fn_substitute_string_with_regex

                    Parameters:
                    source_string:string
                    regex:string
                    replacement:string

                    User:
                    Replace all numbers in \
                        "Hello 34 I'm 233 years old" with NUMBERS

                    Output:
                    source_string:Hello 34 I'm 233 years old,
                    regex:'\\d+',
                    replacement:NUMBERS

                    Function:
                    fn_substitute_string_with_regex

                    Parameters:
                    source_string:string
                    regex:string
                    replacement:string

                    User:
                    Replace all vowels in 'Programming is fun' with asterisks

                    Output:
                    source_string:Programming is fun,
                    regex:[aeiouAEIOU]+,
                    replacement:'*'

                    Function:
                    fn_substitute_string_with_regex

                    Parameters:
                    source_string:string
                    regex:string
                    replacement:string

                    User:
                    Substitute the word 'cat' with 'dog' in \
                        'The cat sat on the mat with another cat'

                    Output:
                    source_string:The cat sat on the mat with another cat,
                    regex:cat,
                    replacement:dog

                    User:
                    {prompt}

                    Output:
                    """

            LEAK_MARKERS = [
                "Function:",
                "Parameters:",
                "Rules",
                "Examples",
                "User:",
                "Output:",
                "(with",
                " (",
            ]
            extracted = ""
            params_dict_str = {}
            for idx, (p_name, p_info) in enumerate(parameter):
                p_type = p_info.type
                param_prompt += f"{p_name}:"
                new = self.extractparam(param_prompt, p_type)
                str_val: str = self.model.decode(new).strip()

                cut_points = [
                    f"{n2}:" for n2, _ in parameter[idx + 1:]
                ] + LEAK_MARKERS
                for marker in cut_points:
                    if marker in str_val:
                        str_val = str_val.split(marker)[0]
                if "," in str_val:
                    str_val = str_val.split(",")[0]
                str_val = str_val.strip().rstrip(",").strip()

                param_prompt += f"{str_val}"
                if idx < len(parameter) - 1:
                    param_prompt += ",\n"
                params_dict_str[p_name] = str_val
                extracted += f"{p_name}:{str_val}" + \
                    (", " if idx < len(parameter) - 1 else "")

            results.append({
                "prompt": prompt,
                "name": data[func_id].name,
                "parameters": params_dict_str
            })

            self.generate_output_file(results)
        return results

    def generate_output_file(
        self, results: list[dict[str, Any]]
    ) -> None:
        BASE_DIR = Path(__file__).resolve().parent.parent
        output_path = \
            BASE_DIR / "data" / "output" / "function_calling_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)


def main() -> None:
    try:
        p = LLM()
        start_time = time.perf_counter()
        prompt = Path("data/input/function_calling_tests.json")
        func = Path("data/input/functions_definition.json")
        print("=== model loaded ===")
        p.processing(prompt, func)
        end_time = time.perf_counter()

        elapsed = end_time - start_time

        print(f"Execution time: {elapsed / 60:.2f} minutes")

    except BaseException as f:
        print(f"ERROR: {f}")


if __name__ == "__main__":
    main()
