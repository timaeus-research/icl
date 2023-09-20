# Initialize counters
files_already_in_destination=0
files_to_be_copied=0

# Loop through each file in the source directory
for src_file in $(aws s3 ls s3://devinterp/checkpoints/icl/ --recursive | awk '{print $4}'); do
  # Remove the source prefix to match against the destination
  dest_file=${src_file#checkpoints/icl/}
  
  # Check if file already exists in the destination
  if grep -q "$dest_file" existing_files.txt; then
    files_already_in_destination=$((files_already_in_destination + 1))
  else
    files_to_be_copied=$((files_to_be_copied + 1))
  fi
done

# Print out the counts
echo "Files already in destination: $files_already_in_destination"
echo "Files to be copied: $files_to_be_copied"
