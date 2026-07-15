from llm_sdk.llm_sdk import Small_LLM_Model 
from pathlib import Path
from pydantic import BaseModel, Field
from parsing import Parsing
from numpy import array, argmax
from math import inf


prompt = Path("data/input/function_calling_tests.json")
func = Path("data/input/functions_definition.json")


class LLM(BaseModel):
    model = Small_LLM_Model()

    def set_param(param: dict):
        pass

    def cons_dec(arr : list):

        numb = LLM.model.encode("0123456789-").squeeze().tolist()
        for i in range(len(arr)):
            if i not in numb:
                arr[i] = -inf

        return arr

    def create_prompt(prompt_path, func_path):
        prompts = Parsing.valid_prompt(prompt_path)
        pars = Parsing()
        data = pars.set_id()

        general_prompt = (
            "you are a function name id selecter "
                "you will seletect a function name depending in user "
                "txt the only way you must do to select "
                "the function name it's by checking name and "
                "description and parameters of the function "
                "to see if it match the user txt, the parameters"
                "of the function must match the one in the "
                f"prompt in number and type here's where the function name"
                f"and the description and parameters store {data}"
                "when you check parameter in the user txt when selected"
                "function if no match found or number of parameters or "
                "types doesn't match output=-7,"
                "example of no match in parameters user txt = "
                "what is the sum of 0? output=-7, here ft_add_number"
                "takes 2 parameters not 1,"
                "example of no match in paramaters user txt = "
                "what is the sum of a? output=-7 here ft_add_number"
                "takers numbers not characters,"
                "or the user txt match a function but the parameters in"
                "user txt isn't enough like function take 2 numbers and in user"
                "txt has only one, output=-7"
                "for the given function and user txt has different type like"
                "string instead of number must output=-7."
                "here's a valid example of user txt = what the sum of 33 and "
                "4660? output=0,"
                "example user txt = what is the square root of 1662?"
                "output=3"
                "example user txt = Reverse the string 'world' output=2,"
                "example user txt = Greet shrek output=1, "
                'example user txt = Replace all numbers in "Hello 34 I\'m 233 '
                'years old" with NUMBERS output=4, '
                f"user txt = "
            )
        enc_gen = LLM.model.encode(general_prompt).squeeze().tolist()
        for prompt in prompts:
            buffer = enc_gen
            buffer.extend(model.encode(f"{prompt} output=").squeeze().tolist())
            lenght = len(buffer)
            logits = LLM.model.get_logits_from_input_ids(buffer)
            logits = cons_dec(logits)
            n = argmax(logits)
            buffer.append(n)
            
            print(model.decode(buffer[lenght:]))

try :
    test = LLM.create_prompt(prompt, func)
except BaseException:
    print(error)
