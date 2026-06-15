from llm_sdk.llm_sdk import Small_LLM_Model
from numpy import argmax, array
def main():
    model = Small_LLM_Model()
    prompt = "am i good person?"
    enc = model.encode(prompt).squeeze().tolist()
    l = len(enc)
    buffer = array("[]")
    for i in range(50):
        buffer = model.get_logits_from_input_ids(enc)
        n = argmax(buffer)
        enc.append(n)
        

    print(model.decode(enc[l:]))
main()