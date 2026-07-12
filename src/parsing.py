from loader import load_prompt, load_funct
from pydantic import BaseModel, Field
import sys
import json
import re

# prompt = Path("data/input/function_calling_tests.json")

# func = load_funct()


class Parsing(BaseModel):

    # def load_prompt():
    #     with prompt.open('r', encoding='utf-8') as f:
    #         return json.load(f)


    def valid_prompt(self):
        with open("data/input/function_calling_tests.json") as f:
            prompts = json.load(f)

        if not isinstance(prompts, list):
            print("ERROR: you didn't load the prompts as a list")

        if len(prompts) == 0:
            print("ERROR: function_calling_tests.json file is empty")
            sys.exit(1)
        for prompt in prompts:
            # print(len(prompt))
            if not isinstance(prompt, dict) :
                print(f"ERROR: {prompt} is not a dict!")
                sys.exit(1)

            if not len(prompt) == 1:
                print(f"ERROR: the dict is too loooong")
                sys.exit(1)

            if "prompt" not in prompt.keys():
                print(f"ERROR: there's a mistake in key here => {prompt}")
                sys.exit(1)

            if not isinstance(prompt["prompt"], str):
                print(f"ERROR: the dict {prompt.values()} should be str")
                sys.exit(1)

            numbers = re.findall(r"\d+", prompt["prompt"])
            # print(numbers)
            for num in numbers:
                num = int(num)
                if num > 2147483647:
                    print(f"ERROR: this number: {num} - is too big try small number")
                    sys.exit(1)

        return prompts


    def valid_function_def(self):

        with open("data/input/functions_definition.json", 'r') as z:
            data = json.load(z)
        
        if not isinstance(data, list):
            print("ERROR: data should be a list")
            sys.exit(1)
        
        if len(data) == 0:
            print("ERROR: the functions_definition.json file is empty")
            sys.exit(1)
        
        for da in data:
            if not isinstance(da, dict):
                print(f"ERROR: {da} is not a dict")
                sys.exit(1)

            if "name" not in da.keys():
                print(f"ERROR: there's no 'name' key here => {da}")
                sys.exit(1)

            if "description" not in da.keys():
                print(f"ERROR: there's no 'description' key here => {da}")
                sys.exit(1)

            if "parameters" not in da.keys():
                print(f"ERROR: there's no 'parameters' key here => {da}")
                sys.exit(1)

            if "returns" not in da.keys():
                print(f"ERROR: there's no 'returns' key here => {da}")
                sys.exit(1)

            if not isinstance(da["name"], str):
                print(f"ERROR: the name '{da["name"]}' \
                    should be a str not {type(da["name"]).__name__}")
                sys.exit(1)

            if not isinstance(da["description"], str):
                print(f"ERROR: in description: '{da["description"]}' \
                    should be a str not {type(da["description"]).__name__}")
                sys.exit(1)

            if not isinstance(da["parameters"], dict):
                print(f"ERROR: in {da} => the parameters key should \
                    be a dict not '{type(da["parameters"]).__name__}")
                sys.exit(1)

            keys = [i for i in da["parameters"].keys()]
            for i in list(keys):
                if not isinstance(da["parameters"][str(i)], dict):
                    print(f"ERROR: {da["parameters"][i]} \
                        should be a dict not {type(da["parameters"][i])}")
                    sys.exit(1)

            if not isinstance(da["returns"], dict):
                print(f"ERROR: the returns key should \
                    be dict not '{type(da["returns"]).__name__}'")







def main():
    test = Parsing()
    test.valid_function_def()
    # print(test1)

if __name__ == "__main__":
    main()