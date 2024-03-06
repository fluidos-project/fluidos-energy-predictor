FROM python:3.11.4

ARG GIT_TAG
LABEL git_tag="$GIT_TAG"
MAINTAINER Matteo Franzil

COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/app"
ENTRYPOINT ["python3", "src/main.py"]