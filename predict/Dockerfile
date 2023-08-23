FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# https://github.com/NVIDIA/nvidia-docker/issues/1009#issuecomment-1181312052
RUN rm /etc/apt/sources.list.d/cuda.list

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# install pytorch and cuda
ARG PYTORCH='2.0.1'
ARG VERSION='torch=='$PYTORCH'.*'
# Example: `cu102`, `cu113`, 'nightly/cu121', etc. ctransformers needs cuda 12+
ARG CUDA='nightly/cu121'
RUN python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA

# avoid gunicorn not found errors
ENV PYTHONPATH=/app 

RUN pip3 install uvicorn[standard]==0.20.0 gunicorn==20.1.0 fastapi[all]==0.88.0

# install ctransformers
RUN pip3 install scikit-build ctransformers 

COPY ./app /app
COPY ./llama2-7b-chat-ggml/ /llama2-7b-chat-ggml

EXPOSE 7080

# Start the app
CMD ["python3", "-m", "gunicorn", "-b", "0.0.0.0:7080", "main:app","--workers","1","--timeout","180","-k","uvicorn.workers.UvicornWorker"]
