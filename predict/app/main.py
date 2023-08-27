""" 
    FastAPI app with the Uvicorn server
"""
from fastapi import FastAPI, Request
from fastapi.logger import logger

from typing import Dict, List, Any
from ctransformers import AutoModelForCausalLM

import json
import logging
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
    
    body = await request.json()  # {'instances': [['who are you ?']]}
    logger.info(f"Body: {body}")

    instances = body["instances"]  # [['who are you ?']]
    logger.info(f"Instances: {instances}")

    outputs = []
    outputs = llm(instances[0][0], max_new_tokens=256, stream=False)
    logger.info(f"Outputs: {outputs}")

    return {"predictions": [outputs]}

