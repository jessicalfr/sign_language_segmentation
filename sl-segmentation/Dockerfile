FROM tensorflow/tensorflow:2.9.2-gpu

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt /requirements.txt

RUN apt-get update && \
    apt-get install --no-install-recommends -y python3 python3-dev python3-pip g++ ffmpeg git

RUN pip install -r requirements.txt

WORKDIR /

RUN rm requirements.txt
