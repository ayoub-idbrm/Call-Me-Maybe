from loader import load_funct, load_prompt
from llm_sdk.llm_sdk import Small_LLM_Model
from numpy import array, argmax

def pars_prompt():
    model = Small_LLM_Model()
    prompts = load_prompt()
    test = prompts[0]['prompt']
    encode = model.encode(test).squeeze()
    lenght = len(encode)
    buff = array("[]")
    for i in range (6):
        buff= model.get_logits_from_input_ids(encode)
        n = argmax(buff)
        encode.append(n)
    print(model.decode(encode[lenght:]))

pars_prompt()