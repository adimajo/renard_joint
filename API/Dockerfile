# pull official base image
FROM python:3.8.3-alpine

# set arguments
ARG the_proxy
ARG PORT
ARG DATA
ARG MODEL

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV HTTP_PROXY=$the_proxy
ENV HTTP_PROXY_AUTH=basic:*::
ENV PORT=$PORT
ENV DATA=$DATA
ENV MODEL=$MODEL

# copy project
COPY . .

# copy data
COPY $DATA /data
ENV DATA=/data

# copy model
COPY $MODEL /model
ENV MODEL=/model

# install dependencies
RUN echo $HTTP_PROXY
RUN echo $HTTP_PROXY_AUTH
RUN http_proxy=$the_proxy && https_proxy=$the_proxy && apk update
RUN apk add --no-cache libgfortran lapack-dev
RUN apk add --no-cache libblas3 libblas-dev libatlas-base-dev
RUN pip install --upgrade pip
RUN pip install gunicorn
RUN pip install renard_joint-0.0.1-py3-none-any.whl

CMD gunicorn wsgi:app --bind 0.0.0.0:8000
