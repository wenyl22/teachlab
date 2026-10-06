# SE-Bench's code sandbox (github.com/thunlp/SE-Bench, Dockerfile), unchanged except that packages come from
# the default PyPI index instead of the Tsinghua mirror. worker.py is SE-Bench's src/evaluation/worker.py.
FROM python:3.11-slim

RUN pip install flask

WORKDIR /sandbox

COPY zwc /sandbox/zwc

RUN cd zwc && \
    pip install -e .

COPY worker.py /sandbox/worker.py

CMD ["python", "worker.py"]
