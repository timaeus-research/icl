#!/bin/bash

TPU_NAME=${1:-forms-tpu}
ZONE_FULL=${2:-us-central1-f}

# Delete the TPU VM
gcloud compute tpus tpu-vm delete $TPU_NAME \
  --zone=$ZONE_FULL
