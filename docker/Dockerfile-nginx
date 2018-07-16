FROM nginx:latest

RUN apt-get update && apt-get install -y \
  curl
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && rm -rf ~/.cache/pip
