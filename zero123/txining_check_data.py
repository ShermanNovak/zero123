import os
import numpy as np
from tqdm import tqdm
import json

root_dir = "/txining/views_release"  # replace with self.root_dir
total_view = 12             # or whatever self.total_view is

with open('/txining/zero123/objaverse-rendering/view_release/valid_paths.json') as f:
    paths = json.load(f)

    for index in tqdm(range(len(paths))):
        folder_path = os.path.join(root_dir, paths[index])
        
        for view_idx in range(total_view):
            file_path = os.path.join(folder_path, f"{view_idx:03d}.npy")

            try:
                _ = np.load(file_path)
            except EOFError:
                print(f"❌ EOFError in: {file_path}")
            except FileNotFoundError:
                print(f"❌ File not found: {file_path}")
            except Exception as e:
                print(f"⚠️ Other error in {file_path}: {type(e).__name__}: {e}")