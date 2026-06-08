# ISLES-26 submission — 5-fold nnUNet ResEncL + ANTs affine MNI registration.
# Weights downloaded from GitHub Release at build time; container is offline at inference.
FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV nnUNet_results=/opt/app/resources/nnUNet_results
ENV nnUNet_raw=/tmp/nnUNet_raw
ENV nnUNet_preprocessed=/tmp/nnUNet_preprocessed
ENV MKL_THREADING_LAYER=GNU
# antspyx downloads template data on first use — pre-cache at build time
ENV ANTSPATH=/opt/app/resources/ants

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --no-cache-dir -r /opt/app/requirements.txt

# Pre-download antspyx data so inference has no network
RUN python -c "import ants; ants.get_ants_data('mni')" || true

# Download MNI152 T1 1mm template (from FSL via GitHub, ~8MB)
ARG MNI_URL=https://github.com/YixinChen-AI/isles26-weights/releases/download/weights-v1/MNI152_T1_1mm.nii.gz
RUN mkdir -p /opt/app/resources && \
    curl --retry 6 --retry-delay 5 --retry-all-errors -fSL \
        "${MNI_URL}" -o /opt/app/resources/MNI152_T1_1mm.nii.gz

# Download 5-fold nnUNet weights from GitHub Release (~782MB per fold)
ARG WEIGHTS_BASE=https://github.com/YixinChen-AI/isles26-weights/releases/download/weights-v1
RUN mkdir -p /opt/app/resources/nnUNet_results /tmp/w && cd /tmp/w \
    && for f in $(seq 0 4); do \
         curl --retry 6 --retry-delay 5 --retry-all-errors -fSL -O "${WEIGHTS_BASE}/fold${f}.tar"; \
       done \
    && for f in $(seq 0 4); do \
         tar xf fold${f}.tar -C /opt/app/resources/nnUNet_results; \
       done \
    && rm -rf /tmp/w \
    && chown -R user:user /opt/app/resources

COPY --chown=user:user inference.py /opt/app/

USER user
ENTRYPOINT ["python", "inference.py"]
