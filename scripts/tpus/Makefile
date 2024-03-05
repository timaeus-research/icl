TPU_NAME ?= forms-tpu
ZONE ?= us-central2-b
ACCELERATOR_TYPE ?= v2-8

setup-tpu:
	./scripts/setup_tpu.sh $(TPU_NAME) $(ZONE) $(ACCELERATOR_TYPE)

train:
	./scripts/train_tpu.sh $(TPU_NAME) $(ZONE)

cleanup:
	./scripts/cleanup_tpu.sh $(TPU_NAME) $(ZONE)

container:
	./scripts/build_container.sh
