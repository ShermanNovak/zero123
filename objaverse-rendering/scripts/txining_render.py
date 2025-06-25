import json
import subprocess
import os

START_INDEX = 0
NUM_VIEWS = 12
GPUINFO = 2
OUTPUT_DIR = "/txining/zero123/objaverse-rendering/fixed"
CAMERA_TYPE = 'fixed'

# Load input JSON
with open("input_models_path.json", "r") as f:
    model_paths = json.load(f)  # expects a list of URLs

# Path to Blender and script
blender_path = "blender-3.2.2-linux-x64/blender"
script_path = "scripts/blender_script.py"

# export valid_paths.json
# ids = [url.split("/")[-1].replace(".glb", "") for url in model_paths]
# with open("valid_paths.json", "w") as f:
#     json.dump(ids, f, indent=2)
# print("Saved", len(ids), "IDs to objaverse_ids.json")

# image extensions
extensions = ['.jpg', '.jpeg', '.png']

# Run command for each model
for model_url in model_paths[START_INDEX:]:

    # skip folders that are completed
    id = model_url.split("/")[-1].replace(".glb", "")
    folder_path = os.path.join(OUTPUT_DIR, id)
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        images = [f for f in files if os.path.splitext(f)[1].lower() in extensions]
        if len(images) == NUM_VIEWS:
            print(f"Skipping {folder_path}")
            continue

    command = [
        f"CUDA_VISIBLE_DEVICES={GPUINFO}",
        blender_path,
        "-b",
        "-P", script_path,
        "--",
        "--object_path", model_url,
        "--camera_type", CAMERA_TYPE,
        "--output_dir", OUTPUT_DIR
    ]

    print(f"Running command for: {model_url}")
    # Join into a single shell command
    full_command = " ".join(command)
    
    try:
        subprocess.run(full_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while processing {model_url}: {e}")
