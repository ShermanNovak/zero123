import os
import json
from tqdm import tqdm

# Path to your main folder
root_dir = "/txining/views_release"

# Initialize empty list for IDs
ids = []

# Iterate through entries in the root directory with a progress bar
for name in tqdm(os.listdir(root_dir), desc="Scanning folders"):
    full_path = os.path.join(root_dir, name)
    if os.path.isdir(full_path):
        ids.append(name)

# Output JSON file path
output_file = "/txining/zero123/objaverse-rendering/view_release/valid_paths_388k.json"

# Write to JSON
with open(output_file, 'w') as f:
    json.dump(ids, f, indent=2)

print(f"Saved {len(ids)} IDs to {output_file}")
