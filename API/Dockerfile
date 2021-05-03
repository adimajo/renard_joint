# pull official base image
FROM python:3.8.3-alpine

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV HTTP_PROXY=http://host.docker.internal:3128
ENV HTTP_PROXY_AUTH=basic:*::

# copy project
COPY . .
COPY . .

# install dependencies
RUN echo $HTTP_PROXY
RUN echo $HTTP_PROXY_AUTH
RUN apk update
RUN apk add --no-cache libgfortran lapack-dev
RUN apk add --no-cache libblas3 libblas-dev libatlas-base-dev
RUN pip install --upgrade pip
RUN pip install pipenv
RUN pipenv install
