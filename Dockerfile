# Start from a PyTorch image optimized for TPU VMs
# FROM gcr.io/tpu-pytorch/xla:r2.0_3.8_tpuvm
FROM pytorch/pytorch

# Set the working directory in the container
WORKDIR /usr/src/app

# Install any necessary dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project into the container
COPY . .

# Use the argument to perform wandb login during image build
RUN echo $WANDB_API_KEY | wandb login

# Command to run the training script
# Note: Adjust the command below according to your specific training script and parameters
# RUN chmod +x ./scripts/wandb/autosweep.sh
# CMD ["./scripts/wandb/autosweep.sh"]