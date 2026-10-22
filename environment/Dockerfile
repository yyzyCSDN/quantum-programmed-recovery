FROM docker.m.daocloud.io/library/python:3.13.15-slim-bookworm@sha256:00faa2debb87529f9f0764e9491d8ba400a3678976616c3bd7cb193745ac20d1
RUN python -m pip install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple numpy==2.5.4 scipy==1.18.1
RUN useradd --create-home --uid 1000 --shell /bin/bash solver \
    && mkdir -p /app /output /logs/verifier \
    && chown solver:solver /app /output \
    && chmod 0755 /logs /logs/verifier
COPY --chown=solver:solver app/ /app/
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
CMD ["sleep", "infinity"]
