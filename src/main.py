from llm_sdk.llm_sdk import Small_LLM_Model
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from parsing import Parsing
from numpy import array, argmax
from math import inf
import sys
import time

start_time = time.process_time()


prompt = Path("data/input/function_calling_tests.json")
func = Path("data/input/functions_definition.json")


class LLM(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Small_LLM_Model = Small_LLM_Model()

    def set_param(self,param: dict):
        # keys = param["parameters"]
        # lst = []
        # n = []
        # for key in keys:
        #     tmp = key
        #     enc = self.model.encode(key)
        #     lst.append(enc)
        # for i in lst:
        #     n.extend(self.model.decode(i))
        # return n
        return list(param["parameters"].items())



    def cons_dec(self, arr: list):
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

    def cons_param_str(self, arr: list):
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,'\"!?-_")
        
        for i in range(len(arr)):
            text = self.model.decode(i)
            if not text or any (c not in allowed_chars for c in text):
                arr[i] = -inf
        return arr

    def cons_param(self, arr: list):
        numb = set()
        for ch in "-0123456789,.":
            ids = self.model.encode(ch).squeeze().tolist()
            if isinstance(ids, int):
                numb.add(ids)
            else:
                numb.update(ids)
        for i in range(len(arr)):
            if i not in numb:
                arr[i] = -inf
        return arr


    def extractparam(self, prompt, param_type="number"):
            buff = self.model.encode(prompt).squeeze().tolist()
            leen = len(buff)
            stop = self.model.encode(",").squeeze().tolist()

            if isinstance(stop, list):
                stop = stop[0]

            for _ in range(20):
                logits = self.model.get_logits_from_input_ids(buff)

                if param_type == "string":
                    logits = self.cons_param_str(logits)
                else:
                    logits = self.cons_param(logits)

                token = argmax(logits)
                buff.append(argmax(logits))

                if token == stop:
                    break
            return buff[leen:]

    def processing(self,prompt_path, func_path):
        prompts = Parsing.valid_prompt(prompt_path)
        pars = Parsing()
        data = pars.set_id()



        function_list = "\n".join(
            f"{i}: {d['name']} - takes parameters: {list(d['parameters'].keys())}"
            for i, d in enumerate(data)
        )

        general_prompt = (
            "You are a function selector. Read the user text and pick the number "
            "of the function that matches it.\n\n"
            "Here are the available functions:\n"
            f"{function_list}\n\n"
            "Rules:\n"
            "- If no function matches, or the number of parameters or their type "
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


        for prompt in prompts:  
            buffer = list(enc_gen)
            buffer.extend(self.model.encode(f"{prompt} output=").squeeze().tolist())
            lenght = len(buffer)

            logits = self.model.get_logits_from_input_ids(buffer)
            logits = self.cons_dec(logits)
            n = argmax(logits)
            buffer.append(n)
            
            # print(model.decode(buffer[lenght:]))
            try:
                n = int(self.model.decode(n))
                print(n)
            except BaseException:
                print("function not found")
                sys.exit(1)
            if n < 0 or n >= len(data):
                print("no matching function found")
                continue 
            parameter = self.set_param(data[n])
            is_str_func = parameter[0][1]["type"] == "string"

            if is_str_func:
                param_prompt = f"""You are a parameter extraction model.

                    Your only task is to extract the parameters for the selected function.

                    Selected function:
                    {data[n]["name"]}

                    Required parameters:
                    {data[n]["parameters"]}

                    Rules:
                    1. Extract ONLY the required parameters.
                    2. Do NOT answer the user's question.
                    3. Do NOT explain anything.
                    4. Do NOT invent missing values.
                    5. Keep the original value exactly as written whenever possible.
                    6. Strings must be strings without changing their content.
                    7. Output the parameters in the same order as the function definition.
                    8. If a required parameter cannot be found, output MISSING.
                    9. If there are multiple values of the same type, choose the ones that belong to the user's request.
                    10. Output each parameter as parameter_name:value.
                    11. Separate consecutive parameters with ",\n".
                    12. Do not output anything except the extracted parameters.
                    13. Strip surrounding quotes (single or double) from string values, keep the inner content exactly as written.

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
                    Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS

                    Output:
                    source_string:Hello 34 I'm 233 years old,
                    regex:[0-9]+,
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
                    replacement:asterisks

                    Function:
                    fn_substitute_string_with_regex

                    Parameters:
                    source_string:string
                    regex:string
                    replacement:string

                    User:
                    Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'

                    Output:
                    source_string:The cat sat on the mat with another cat,
                    regex:cat,
                    replacement:dog

                    User:
                    {prompt}

                    Output:
                    """
            else :
                param_prompt = f"""You are a parameter extraction model.

                    Your only task is to extract the parameters for the selected function.

                    Selected function:
                    {data[n]["name"]}

                    Required parameters:
                    {data[n]["parameters"]}

                    Rules:
                    1. Extract ONLY the required parameters.
                    2. Do NOT answer the user's question.
                    3. Do NOT explain anything.
                    4. Do NOT invent missing values.
                    5. Keep the original value exactly as written whenever possible.
                    6. Numbers must be numbers (integers or decimals).
                    7. Output the parameters in the same order as the function definition.
                    8. If a required parameter cannot be found, output MISSING.
                    9. If there are multiple numbers in the user text, choose the ones that belong to the user's request, in the order they appear.
                    10. Output each parameter as parameter_name:value.
                    11. Separate consecutive parameters with ",\n".
                    12. Do not output anything except the extracted parameters.
                    13. Decimal numbers must keep their decimal point (e.g. 2.45, not 2 or 245).

                    Examples

                    Function:
                    fn_add_numbers

                    Parameters:
                    a:number
                    b:number

                    User:
                    What is the sum of 23 and 91?

                    Output:
                    a:23,
                    b:91

                    Function:
                    fn_add_numbers

                    Parameters:
                    a:number
                    b:number

                    User:
                    What is the sum of 2.45 and 3?

                    Output:
                    a:2.45,
                    b:3

                    Function:
                    fn_add_numbers

                    Parameters:
                    a:number
                    b:number

                    User:
                    What is the sum of 265 and 345?

                    Output:
                    a:265,
                    b:345

                    Function:
                    fn_get_square_root

                    Parameters:
                    a:number

                    User:
                    Square root of 144

                    Output:
                    a:144

                    Function:
                    fn_get_square_root

                    Parameters:
                    a:number

                    User:
                    What is the square root of 16?

                    Output:
                    a:16

                    User:
                    {prompt}

                    Output: 
                    """

            strr = param_prompt
            buff = []
            buff = self.model.encode(param_prompt).squeeze().tolist()
            # buff.extend(buff).squeeze().tolist()
            leen = len(buff)
            stop = self.model.encode(",\n").squeeze().tolist()


            lenght_prom = len(param_prompt)

            i = 0
            extracted = ""
            for p_name, p_info in parameter:
                p_type = p_info["type"]
                param_prompt += f"{p_name}:"          # show the correct label BEFORE generating
                new = self.extractparam(param_prompt, p_type)
                val = self.model.decode(new)
                param_prompt += f"{val} "
                extracted += f"{p_name}:{val} "
            print(f"prompt: {extracted.strip()}")     
            # print(param_prompt[len(strr):])
            
            





# try :
p = LLM()
test = p.processing(prompt, func)
end_time = time.process_time()
cpu_time = end_time - start_time
print(f"CPU time: {cpu_time} seconds")
# except BaseException as e:
#     print("error",e)
