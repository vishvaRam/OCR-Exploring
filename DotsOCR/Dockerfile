FROM python:3.12.14-slim-bookworm

RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|http://|https://|g' /etc/apt/sources.list.d/debian.sources; \
    elif [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -i 's|http://|https://|g' /etc/apt/sources.list.d/ubuntu.sources; \
    elif [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http://|https://|g' /etc/apt/sources.list; \
    fi
    
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip --break-system-packages

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

CMD ["/bin/bash"]