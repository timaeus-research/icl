# List all existing files in the destination
echo "Listing existing files in the destination..."
aws s3 ls s3://devinterp/backups/icl/ --recursive | awk '{print $4}' > existing_files.txt
echo "Done."

# Loop through each file in the source directory
echo "Looping through each file in the source directory..."
for src_file in $(aws s3 ls s3://devinterp/checkpoints/icl/ --recursive | awk '{print $4}'); do
  # Remove the source prefix to match against the destination
  dest_file=${src_file#checkpoints/icl/}
  
  # Check if file already exists in the destination
  if ! grep -q "$dest_file" existing_files.txt; then
    # If not, copy the file
    echo "Copying $src_file to $dest_file..."
    aws s3 cp "s3://devinterp/$src_file" "s3://devinterp/backups/icl/$dest_file"
  fi
done
echo "Done."

# Clean up
rm existing_files.txt
