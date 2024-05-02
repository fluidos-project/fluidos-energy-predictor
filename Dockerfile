FROM python:3.11.4

ARG GIT_TAG
LABEL git_tag="$GIT_TAG"
MAINTAINER Matteo Franzil

COPY . /app
WORKDIR /app

# Install necessary dependencies for building h5py
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone h5py and modify setup.py for compatibility
RUN git clone https://github.com/h5py/h5py.git && \
    cd h5py && \
    sed -i 's/1.19.3/1.20.1/g' setup.py && \
    cd ..

# Install h5py from source
RUN pip install ./h5py
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/app"
ENTRYPOINT ["python3", "src/main.py"]