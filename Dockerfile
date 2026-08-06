ARG BASE_IMAGE=pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime
FROM --platform=linux/amd64 ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    nnUNet_compile=0 \
    nnUNet_raw=/tmp/nnUNet_raw \
    nnUNet_preprocessed=/tmp/nnUNet_preprocessed \
    nnUNet_results=/opt/ml/model \
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

COPY --chown=user:user app.py inference.py /opt/app/

USER user
LABEL org.grand-challenge.api-method="invoke"
EXPOSE 4743
ENTRYPOINT ["python", "app.py"]
