Here the model to be downloaded from [Hugging Face](https://huggingface.co/TheBloke/Dolphin-Llama2-7B-GGML/tree/main). 
No `handler.py` required since we will not use TorchServe. Note also the size (2 GiB since it is a 2-bit GGML model):
```sh
config.json
dolphin-llama2-7b.ggmlv3.q2_K.bin
```