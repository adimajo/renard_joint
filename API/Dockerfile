# pull official base image
FROM python:3.8.3-alpine

# set environment variables
ARG the_proxy
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV HTTP_PROXY=$the_prox
ENV HTTP_PROXY_AUTH=basic:*::

# copy project
COPY . .

# install dependencies
RUN echo $HTTP_PROXY
RUN echo $HTTP_PROXY_AUTH
RUN apk update
RUN apk add --no-cache libgfortran lapack-dev
RUN apk add --no-cache libblas3 libblas-dev libatlas-base-dev
RUN pip install --upgrade pip
RUN pip install renard_joint-0.0.1-py3-none-any.whl

ENTRYPOINT [ "python" ]

CMD [ "wsgi.py" ]
