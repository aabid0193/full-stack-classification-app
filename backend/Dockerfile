FROM ubuntu:18.04

RUN apt-get update \
    && apt-get install -yq --no-install-recommends \
    python3-dev \
    python3-pip \
    build-essential
# when upgrading pip, pip is used instead of pip3
RUN pip3 install --upgrade pip \ 
    && pip install setuptools

EXPOSE 8081

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt 
RUN [ "python3", "-c", "import nltk; nltk.download('punkt')" ]

ENTRYPOINT ["python3"]
CMD ["app.py"]