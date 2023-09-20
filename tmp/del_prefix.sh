# List all object versions in the 'backups/' prefix
aws s3api list-object-versions --bucket devinterp --prefix backups/ --output json |
jq -r '.Versions[] | .Key + " " + .VersionId' > to_delete.txt

# Delete each version
while read -r line; do
  KEY=$(echo $line | cut -d' ' -f1)
  VERSION_ID=$(echo $line | cut -d' ' -f2)
  aws s3api delete-object --bucket devinterp --key "$KEY" --version-id "$VERSION_ID"
done < to_delete.txt

# Clean up
rm to_delete.txt
