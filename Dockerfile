FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set work directory
WORKDIR /workspace

# Install general-purpose dependencies
RUN apt-get update -y && \
	apt-get install -y curl \
	git \
	bash \
	nano \
	wget \
	python3.9 \
	python3-pip && \
	apt-get autoremove -y && \
	apt-get clean -y && \
	rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /workspace
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
