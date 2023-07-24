# image python:3.8.5-slim-buster
FROM pytorch/pytorch:latest

RUN apt update
RUN apt install -y build-essential
RUN pip install --upgrade pip

RUN pip install tensorboard einops lightning pandas scipy scikit-learn matplotlib seaborn
