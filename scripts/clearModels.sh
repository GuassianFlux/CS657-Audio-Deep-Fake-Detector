folder="../trained_models"

# Check if the specified path exists and is a directory
if [ ! -d "$folder" ]; then
    echo "Error: '$folder' is not a directory or doesn't exist."
    exit 1
fi

# Prompt the user for confirmation before proceeding
read -p "Are you sure you want to delete all files and folders in '$folder'? (y/n): " answer

if [ "$answer" != "y" ]; then
    echo "Aborted."
    exit 0
fi

# Delete all files and folders within the specified directory
rm -rf "$folder"/*

echo "All files and folders in '$folder' have been deleted."