# The environment in which `uv run fab docling-fixtures` converts the fixtures in
# Docker (`--docker`, the default on Windows), on Linux as the workflow
# docling-fixtures.yml does: Ubuntu with LibreOffice and ffmpeg, as the workflow
# installs them, and the uv of the workflows, whose version dev/docling_fixtures.py
# passes (UV_VERSION). It copies no file of the checkout, which is mounted at /repo
ARG UV_VERSION
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ubuntu:24.04
# LibreOffice (Writer, Calc and Impress), with which Docling converts the legacy
# Office and RTF fixtures (DOC, XLS, PPT, RTF), and ffmpeg, with which Whisper
# reads the audio and video fixtures and Docling the frames of a video, without
# the packages that they only recommend
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates ffmpeg libreoffice-calc libreoffice-impress \
        libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
# What the runs keep is in the volume mounted at /cache: uv's cache and Pythons,
# the environment of the project, which is not the .venv of the mounted checkout,
# that of the host, and the models of Docling and Whisper (under XDG_CACHE_HOME and
# HOME). No bytecode is written into the checkout
ENV HOME=/cache/home \
    XDG_CACHE_HOME=/cache \
    UV_CACHE_DIR=/cache/uv \
    UV_PYTHON_INSTALL_DIR=/cache/uv-python \
    UV_PROJECT_ENVIRONMENT=/cache/contextgem-venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HUB_DOWNLOAD_TIMEOUT=30
WORKDIR /repo
