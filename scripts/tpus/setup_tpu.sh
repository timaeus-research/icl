#!/bin/bash

TPU_NAME=${1:-forms-tpu}
ZONE=${2:-us-central1}
ZONE_FULL=${2:-us-central1-f}
ACCELERATOR_TYPE=${3:-v4-32}
PROJECT_ID="liquid-braid-400023"

DOCKER_ENDPOINT="$ZONE-docker.pkg.dev"
DOCKER_REPO="timaeus-$ZONE"
DOCKER_IMAGE_NAME="forms"
DOCKER_TAG="latest"
DOCKER_PATH="$DOCKER_ENDPOINT/$PROJECT_ID/$DOCKER_REPO/$DOCKER_IMAGE_NAME:$DOCKER_TAG"

# Create TPU VM with optional arguments for zone and accelerator type
gcloud compute tpus tpu-vm create ${TPU_NAME} \
    --zone=${ZONE_FULL} \
    --accelerator-type=${ACCELERATOR_TYPE} \
    --version=tpu-ubuntu2204-base

# Add the current user to the docker group
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE_FULL} \
    --worker=all \
    --command="sudo usermod -a -G docker \$USER"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} --worker=all \
    --zone=${ZONE_FULL} \
    --command="gcloud auth configure-docker ${DOCKER_REGISTRY} --quiet" 

gcloud compute tpus tpu-vm ssh ${TPU_NAME} --worker=all \
    --zone=europe-west4-a \
    --command="docker pull $DOCKER_IMAGE_NAME $DOCKER_PATH"
