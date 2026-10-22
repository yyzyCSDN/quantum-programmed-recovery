FROM public.ecr.aws/docker/library/python:3.13.15-slim-bookworm@sha256:00faa2debb87529f9f0764e9491d8ba400a3678976616c3bd7cb193745ac20d1
COPY requirements.txt /tmp/requirements.txt
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG PIP_TRUSTED_HOST=
RUN python -m pip install --index-url "$PIP_INDEX_URL" --no-cache-dir --only-binary=:all: --require-hashes -r /tmp/requirements.txt
RUN useradd --create-home --uid 1000 --shell /bin/bash solver \
    && mkdir -p /app /output /logs/verifier \
    && chown solver:solver /app /output \
    && chmod 0755 /logs /logs/verifier
COPY --chown=solver:solver app/ /app/
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
CMD ["sleep", "infinity"]
