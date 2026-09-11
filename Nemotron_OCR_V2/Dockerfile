FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -i 's|http://|https://|g' /etc/apt/sources.list.d/ubuntu.sources; \
    elif [ -f /etc/apt/sources.list.d/ubuntu.source ]; then \
        sed -i 's|http://|https://|g' /etc/apt/sources.list.d/ubuntu.source; \
    elif [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http://|https://|g' /etc/apt/sources.list; \
    fi

# -----------------------------------------------------------------------------
# 1. Install ALL Build Dependencies for Full Python Support
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic build tools
    build-essential \
    wget \
    curl \
    ca-certificates \
    gnupg \
    pkg-config \
    # Compression (Fixes: _lzma, _bz2, zlib)
    liblzma-dev \
    libbz2-dev \
    zlib1g-dev \
    # Crypto & SSL (Fixes: _ssl, _hashlib)
    libssl-dev \
    libnss3-dev \
    # Database (Fixes: _sqlite3, _dbm, _gdbm)
    libsqlite3-dev \
    libgdbm-dev \
    libgdbm-compat-dev \
    # Terminal & UI (Fixes: _curses, _readline, _tkinter)
    libreadline-dev \
    libncursesw5-dev \
    tk-dev \
    # XML & System (Fixes: _ctypes, _uuid, pyexpat)
    libffi-dev \
    uuid-dev \
    libxml2-dev \
    libxmlsec1-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. Build and Install Python 3.12.14
# -----------------------------------------------------------------------------
ARG PY_VER=3.12.14
ARG PY_URL=https://www.python.org/ftp/python/${PY_VER}/Python-${PY_VER}.tgz

RUN wget ${PY_URL} && \
    tar -xf Python-${PY_VER}.tgz && \
    cd Python-${PY_VER} && \
    # --enable-shared: builds libpython3.12.so (required by some ML libs like vLLM/PyTorch)
    # --with-ensurepip: installs pip automatically
    ./configure \
        --enable-optimizations \
        --enable-shared \
        --with-ensurepip=install \
        LDFLAGS="-Wl,-rpath=/usr/local/lib" && \
    make -j"$(nproc)" && \
    make altinstall && \
    ldconfig && \
    cd / && \
    rm -rf Python-${PY_VER}*

# -----------------------------------------------------------------------------
# 3. Configure Global Defaults (No venv, System Python)
# -----------------------------------------------------------------------------
# Point system 'python' and 'pip' to our new 3.12 install
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/local/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/pip3    pip3    /usr/local/bin/pip3.12    1 && \
    update-alternatives --install /usr/bin/pip     pip     /usr/local/bin/pip3.12    1

# Upgrade the global pip to the latest version
RUN python -m pip install --upgrade pip setuptools wheel

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        build-essential \
        ninja-build \
        libgl1 \
        libglib2.0-0 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130

RUN pip install hatchling

WORKDIR /app/nemotron-ocr-v2/nemotron-ocr
RUN pip install --no-build-isolation -v .

RUN pip install -r requirements.txt

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]