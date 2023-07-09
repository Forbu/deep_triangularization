# image python:3.8.5-slim-buster
FROM python:3.8.5-slim-buster

RUN apt update
RUN apt install -y build-essential
RUN pip install --upgrade pip
