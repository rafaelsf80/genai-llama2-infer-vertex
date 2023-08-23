""" 
    Simple local inference of Llama 2-7B model on CPU using ctransformers
"""

from ctransformers import AutoModelForCausalLM

config = {'max_new_tokens': 256, 'repetition_penalty': 1.1, 'temperature': 0.1, 'stream': True}

# choose your champion
#model_id = "TheBloke/Llama-2-7B-GGML"
#model_id = "TheBloke/Llama-2-7B-chat-GGML"
#model_id = "TheBloke/Llama-2-13B-GGML"
#model_id = "TheBloke/Llama-2-13B-chat-GGML"
model_id = './predict/llama2-7b-chat-ggml/'

llm = AutoModelForCausalLM.from_pretrained(model_id,
                                           model_type="llama",
                                           lib='avx2', #for cpu use
                                           #gpu_layers=130, #110 for 7b, 130 for 13b
                                           **config
                                           )


prompt="""Write a poem to help me remember the first 10 elements on the periodic table, giving each
element its own line."""

tokens = llm.tokenize(prompt)

# 'pipeline' execution
print(llm(prompt, max_new_tokens=256, stream=False))