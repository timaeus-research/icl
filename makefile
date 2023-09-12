xla-max:
	python -m icl
xla-min:
	PJRT_DEVICE="TPU" python tpu-train.py xla
cpu-min:
	PJRT_DEVICE="TPU" python tpu-train.py cpu
