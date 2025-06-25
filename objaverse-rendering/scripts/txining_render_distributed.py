import os
import json
import time
import math
import subprocess
import multiprocessing
from dataclasses import dataclass
from typing import Optional

import tyro
import boto3
import wandb

EXTENSIONS = {'.jpg', '.jpeg', '.png'}

@dataclass
class Args:
    workers_per_gpu: int
    input_models_path: str
    upload_to_s3: bool = False
    log_to_wandb: bool = False
    num_gpus: int = -1  # -1 = use all available


def is_render_complete(view_path: str) -> bool:
    if not os.path.exists(view_path):
        return False
    image_files = [f for f in os.listdir(view_path) if os.path.splitext(f)[1].lower() in EXTENSIONS]
    return len(image_files) >= 6


def worker(queue: multiprocessing.JoinableQueue, count: multiprocessing.Value, gpu: int) -> None:
    while True:
        model_url = queue.get()
        if model_url is None:
            break

        obj_id = os.path.basename(model_url).replace(".glb", "")
        view_path = os.path.join('heuristic', obj_id)

        if is_render_complete(view_path):
            print(f"======== Skipping {obj_id}, already rendered ========")
            queue.task_done()
            continue

        os.makedirs(view_path, exist_ok=True)

        command = (
            f"CUDA_VISIBLE_DEVICES={gpu} "
            f"blender-3.2.2-linux-x64/blender -b -P scripts/blender_script.py --"
            f" --object_path {model_url} --camera_type heuristic --output_dir heuristic"
        )

        print(f"[GPU {gpu}] Rendering: {obj_id}")
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {model_url}: {e}")

        with count.get_lock():
            count.value += 1

        queue.task_done()


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.num_gpus == -1:
        args.num_gpus = torch.cuda.device_count()

    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    if args.log_to_wandb:
        wandb.init(project="objaverse-rendering", entity="prior-ai2")

    # Start worker processes per GPU
    processes = []
    for gpu in range(args.num_gpus):
        for _ in range(args.workers_per_gpu):
            p = multiprocessing.Process(target=worker, args=(queue, count, gpu))
            p.daemon = True
            p.start()
            processes.append(p)

    # Load input JSON (list of URLs)
    with open(args.input_models_path, "r") as f:
        model_urls = json.load(f)

    for model_url in model_urls:
        queue.put(model_url)

    # WandB logging loop
    if args.log_to_wandb:
        while True:
            time.sleep(5)
            wandb.log({
                "completed": count.value,
                "total": len(model_urls),
                "progress": count.value / len(model_urls),
            })
            if count.value >= len(model_urls):
                break

    queue.join()

    # Signal workers to shut down
    for _ in processes:
        queue.put(None)
    for p in processes:
        p.join()
