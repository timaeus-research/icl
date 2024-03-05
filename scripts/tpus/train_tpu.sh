#!/bin/bash

TPU_NAME=${1:-forms-tpu}
ZONE=${2:-us-central1}
ZONE_FULL=${3:-us-central1-f}
PROJECT_ID="liquid-braid-400023"

DOCKER_ENDPOINT="$ZONE-docker.pkg.dev"
DOCKER_REPO="timaeus-$ZONE"
DOCKER_IMAGE_NAME="forms"
DOCKER_TAG="latest"
DOCKER_PATH="$DOCKER_ENDPOINT/$PROJECT_ID/$DOCKER_REPO/$DOCKER_IMAGE_NAME:$DOCKER_TAG"

# Run the training script in a container on all TPU workers

gcloud compute tpus tpu-vm ssh ${TPU_NAME} --worker=all \
    zone=${ZONE_FULL} \
    --command="docker run -ti -d --privileged --net=host --name $DOCKER_CONTAINER_NAME $DOCKER_PATH bash"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} --worker=all \
    --zone=${ZONE_FULL} \
    --command="docker exec --privileged $DOCKER_CONTAINER_NAME ./scripts/wandb/autosweep.sh"