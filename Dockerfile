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
ARG WEIGHTS_BASE=https://github.com/YixinChen-AI/isles26-weights/releases/download/dataset503-dual-ensemble-r1
ARG MODEL_ARCHIVE_SHA256=f351ad8c980da127f491fb2224e63fd4a41659c55bccc5a757280afad64b1639
ARG WEIGHTS_PARTS="model.tar.gz.part-aa model.tar.gz.part-ab model.tar.gz.part-ac model.tar.gz.part-ad"
RUN mkdir -p /opt/app/model \
    && : > /tmp/model.tar.gz \
    && for part in ${WEIGHTS_PARTS}; do \
        case "${part}" in \
          model.tar.gz.part-[a-z][a-z]) ;; \
          *) echo "invalid model archive part: ${part}" >&2; exit 1 ;; \
        esac; \
        curl --http1.1 --continue-at - \
          --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
          --retry 6 --retry-delay 5 --retry-all-errors -fSL \
          -o "/tmp/${part}" "${WEIGHTS_BASE}/${part}"; \
        cat "/tmp/${part}" >> /tmp/model.tar.gz; \
      done \
    && echo "${MODEL_ARCHIVE_SHA256}  /tmp/model.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/model.tar.gz -C /opt/app/model \
    && rm -f /tmp/model.tar.gz /tmp/model.tar.gz.part-??

COPY --chown=user:user app.py inference.py /opt/app/
COPY --chown=user:user isles26_model_manifest.json /opt/app/model/isles26_model_manifest.json

USER user
LABEL org.grand-challenge.api-method="invoke"
EXPOSE 4743
ENTRYPOINT ["python", "app.py"]
