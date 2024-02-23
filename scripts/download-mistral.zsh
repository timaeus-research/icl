#!/bin/zsh

# Step 1: Setup and Configuration
LOCAL_ROOT="./checkpoints"  # Update this path
# AWS_LANGUAGE_BUCKET_NAME="your-aws-bucket-name"  # Load from .env
declare -A models=(
  ["arwen"]="stanford-crfm/arwen-gpt2-medium-x21"
  ["beren"]="stanford-crfm/beren-gpt2-medium-x49"
  ["celebrimbor"]="stanford-crfm/celebrimbor-gpt2-medium-x81"
  ["durin"]="stanford-crfm/durin-gpt2-medium-x343"
  ["eowyn"]="stanford-crfm/eowyn-gpt2-medium-x777"
  ["alias"]="stanford-crfm/alias-gpt2-small-x21"
  ["battlestar"]="stanford-crfm/battlestar-gpt2-small-x49"
  ["caprica"]="stanford-crfm/caprica-gpt2-small-x81"
  ["darkmatter"]="stanford-crfm/darkmatter-gpt2-small-x343"
  ["expanse"]="stanford-crfm/expanse-gpt2-small-x777"
)

# Load .env
source .env
echo "Bucket: $AWS_LANGUAGE_BUCKET_NAME"

# typeset -A models
# models=(
#   arwen "stanford-crfm/arwen-gpt2-medium-x69"
#   beren "stanford-crfm/beren-gpt2-medium-x69"
#   celebrimbor "stanford-crfm/celebrimbor-gpt2-medium-x69"
#   durin "stanford-crfm/durin-gpt2-medium-x69"
#   eowyn "stanford-crfm/eowyn-gpt2-medium-x69"
#   alias "stanford-crfm/alias-gpt2-small-x21"
#   battlestar "stanford-crfm/battlestar-gpt2-small-x49"
#   caprica "stanford-crfm/caprica-gpt2-small-x21"
#   darkmatter "stanford-crfm/darkmatter-gpt2-small-x21"
#   expanse "stanford-crfm/expanse-gpt2-small-x21"
# )

# Generate checkpoints based on the described stepping
# checkpoints=()
# for step in {0..100..10}; do checkpoints+=($step); done
# for step in {150..2000..50}; do checkpoints+=($step); done
# for step in {2100..20000..100}; do checkpoints+=($step); done
# for step in {21000..400000..1000}; do checkpoints+=($step); done
checkpoints=($(seq 10 100 10) $(seq 150 50 2000) $(seq 2100 100 20000) $(seq 21000 1000 400000))

# Step 2: Download, Organize Checkpoints, and Upload
git lfs install

model_type_pattern='^(arwen|beren|celebrimbor|durin|eowyn)$'

# echo "Iterating over ${models[@]}..."
for model repo in "${(@kv)models}"; do
  echo "----------------------------------------"
  echo "Processing $model ($repo)"
  size="small"  # Default size, adjust based on the model
  if [[ $model =~ $model_type_pattern ]]; then
    size="medium"
  fi

  for checkpoint in $checkpoints; do
    # Define local path
    local_path="$LOCAL_ROOT/gpt-2-$size-$model/$checkpoint"
    mkdir -p "$local_path"
    echo "Cloning $model ($size) checkpoint $checkpoint to $local_path"

    # Check if the checkpoint exists
    if [ -d "$local_path" ]; then
      echo "Checkpoint $checkpoint does not exist, cloning..."
      git clone --branch "checkpoint-$checkpoint" --single-branch "https://huggingface.co/$repo" "$local_path"
      (cd "$local_path" && git lfs pull)  
    else
      echo "Checkpoint $checkpoint already exists, skipping..."
    fi
    rm -rf "$local_path/.git"

    # Upload to AWS
    echo "Uploading $model ($size) checkpoint $checkpoint to AWS"
    aws s3 cp "$local_path" "s3://$AWS_LANGUAGE_BUCKET_NAME/checkpoints/gpt-2-$size-$model/$checkpoint/" --recursive

    # Raise error
    # Cleanup: Remove the checkpoint directory after upload
    echo "Cleaning up $local_path"
    rm -rf "$local_path"
  
  done
done
