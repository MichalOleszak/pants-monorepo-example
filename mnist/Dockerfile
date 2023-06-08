FROM python:3.9

ENV PYTHONPATH .
WORKDIR /app

COPY mnist mnist

RUN pip3 install -r mnist/requirements.txt
CMD ["python", "mnist/src/train.py"]
