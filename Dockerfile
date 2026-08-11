ARG BASE_IMAGE=pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime
FROM --platform=linux/amd64 ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    nnUNet_compile=0 \
    nnUNet_raw=/tmp/nnUNet_raw \
    nnUNet_preprocessed=/tmp/nnUNet_preprocessed \
    nnUNet_results=/opt/app/model \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=8

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
WORKDIR /opt/app

RUN python -m venv --system-site-packages --without-pip /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Checkpoints are immutable GitHub Release assets. Submission variants reuse
# these bytes and version only the small policy manifest with the code tag.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
ARG WEIGHTS_BASE=https://github.com/YixinChen-AI/isles26-weights/releases/download/dataset503-baseline-fixed500-r1
ARG MODEL_ARCHIVE_SHA256=3f08f161d9710b8b6b9374f68f5804b16bdb4bacea3c1fd2a9e68b91a54e9d80
RUN mkdir -p /opt/app/model \
    && curl --retry 6 --retry-delay 5 --retry-all-errors -fSL \
        -o /tmp/model.tar.gz.part-aa "${WEIGHTS_BASE}/model.tar.gz.part-aa" \
    && curl --retry 6 --retry-delay 5 --retry-all-errors -fSL \
        -o /tmp/model.tar.gz.part-ab "${WEIGHTS_BASE}/model.tar.gz.part-ab" \
    && cat /tmp/model.tar.gz.part-aa /tmp/model.tar.gz.part-ab > /tmp/model.tar.gz \
    && echo "${MODEL_ARCHIVE_SHA256}  /tmp/model.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/model.tar.gz -C /opt/app/model \
    && rm -f /tmp/model.tar.gz /tmp/model.tar.gz.part-aa /tmp/model.tar.gz.part-ab

COPY --chown=user:user app.py inference.py /opt/app/
COPY --chown=user:user isles26_model_manifest.json /opt/app/model/isles26_model_manifest.json

USER user
LABEL org.grand-challenge.api-method="invoke"
EXPOSE 4743
ENTRYPOINT ["python", "app.py"]
