# pull official base image
FROM python:3.8-slim

# set arguments
ARG the_proxy
ARG the_path
ARG PORT
ARG DATA
ARG MODEL

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV the_proxy=$the_proxy
ENV the_path=$the_path
ENV PORT=$PORT
ENV DATA=$DATA
ENV MODEL=$MODEL

# copy project
COPY $the_path/API .

# copy data
COPY $DATA data
ENV DATA=data

# copy model
COPY $MODEL model
ENV MODEL=model

# install dependencies
RUN http_proxy=$the_proxy && https_proxy=$the_proxy && pip install --upgrade pip
RUN http_proxy=$the_proxy && https_proxy=$the_proxy && pip install gunicorn
RUN http_proxy=$the_proxy && https_proxy=$the_proxy && pip install renard_joint-0.0.1-py3-none-any.whl

CMD gunicorn wsgi:app --bind 0.0.0.0:8000
