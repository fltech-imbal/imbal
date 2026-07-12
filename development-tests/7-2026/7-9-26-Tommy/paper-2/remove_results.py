from pathlib import Path
import os

folder_path = Path("results_onp")
prefix = "v"

# Finds all matching files recursively
matching_files = [
    str(file) for file in folder_path.rglob(f"{prefix}*") if file.is_file()
]

for file in matching_files:
    os.remove(file)