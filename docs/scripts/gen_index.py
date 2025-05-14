import shutil

# Define paths for README.md and docs/index.md
readme_path = "README.md"  # Assuming README.md is in the root directory
index_path = "docs/index.md"

# Copy README.md to docs/index.md
try:
    shutil.copyfile(readme_path, index_path)
    print(f"Successfully copied {readme_path} to {index_path}")
except FileNotFoundError:
    print(f"Error: {readme_path} not found. Make sure it exists in the root directory.")
except Exception as e:
    print(f"An error occurred: {e}")

