""" 
    FastAPI app with the Uvicorn server
"""
from fastapi import FastAPI, Request
from fastapi.logger import logger

from typing import Dict, List, Any
from ctransformers import AutoModelForCausalLM

import json
import logging
#import numpy as np
import os

import torch


app = FastAPI()

gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers

if __name__ != "main":
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.INFO)

logger.info(f"Is CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

config = {'max_new_tokens': 256, 'repetition_penalty': 1.1, 'temperature': 0.1, 'stream': True}

model_id = '../llama2-7b-chat-ggml'

logger.info(f"Loading model {model_id}. This takes some time ...")

llm = AutoModelForCausalLM.from_pretrained(model_id,
                                           model_type="llama",
                                           #lib='avx2', #for cpu use
                                           gpu_layers=110, #110 for 7b, 130 for 13b
                                           **config
                                           )
logger.info(f"Loading model DONE")




@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {"status": "healthy"}

@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):

 #   ERROR 2023-08-18T13:30:43.865566015Z [2023-08-18 13:30:43 +0000] [28] [INFO] Body: {'instances': [['who are you ?']]}
#ERROR 2023-08-18T13:30:43.865620613Z [2023-08-18 13:30:43 +0000] [28] [INFO] Instances: [['who are you ?']]
    
    body = await request.json()  # {'instances': [['who are you ?']]}

    print(body)
    logger.info(f"Body: {body}")


    instances = body["instances"]  # [['who are you ?']]
    logger.info(f"Instances: {instances}")


    print(instances)

    #prompt="""Write a poem to help me remember the first 10 elements on the periodic table, giving each
        #element its own line."""

    #tokens = llm.tokenize(instances)

    outputs = []
    outputs = llm(instances[0][0], max_new_tokens=256, stream=False)

    print(outputs)
    logger.info(f"Outputs: {outputs}")


#    for instance in instances:
        #input_ids = tokenizer(instance, return_tensors="pt").input_ids

        # 'pipeline' execution


        #output = model.generate(input_ids)
        #prediction = tokenizer.decode(output[0], skip_special_tokens=True)
        #outputs.append(prediction)


    return {"predictions": [outputs]}

