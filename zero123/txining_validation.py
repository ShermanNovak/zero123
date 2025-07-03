import os
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import webdataset as wds
from torchvision import transforms
import torchvision
from einops import rearrange

from ldm.util import instantiate_from_config
from ldm.modules.evaluate.evaluate import compute_evaluation_metrics
from ldm.data.simple import ObjaverseDataModuleFromConfig, ObjaverseData

# 1. Load config and instantiate model (adapt path as needed)
config_path = "/txining/zero123/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml"  # <-- set this
ckpt_path = "/txining/zero123/zero123/105000.ckpt"  # <-- set this

config = OmegaConf.load(config_path)
model = instantiate_from_config(config.model)
state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
model.load_state_dict(state_dict, strict=False)
model.cuda()
model.eval()

# 2. Instantiate the datamodule
data_cfg = config.data
# datamodule = ObjaverseDataModuleFromConfig(**data_cfg.params)
# datamodule.prepare_data()
# datamodule.setup()
# test_loader = datamodule.test_dataloader()

image_transforms = [torchvision.transforms.Resize(data_cfg.params.validation.image_transforms.size)]
image_transforms.extend([transforms.ToTensor(),
                        transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
image_transforms = torchvision.transforms.Compose(image_transforms)

val_dataloader = wds.WebLoader(ObjaverseData(root_dir=data_cfg.params.root_dir, total_view=data_cfg.params.total_view, validation=True, test=False, \
                                image_transforms=image_transforms), batch_size=data_cfg.params.batch_size, num_workers=data_cfg.params.num_workers, shuffle=False)

# 3. Evaluate with tqdm
all_metrics = []
for batch in tqdm(val_dataloader, desc="Evaluating"):
    with torch.no_grad():
        device = next(model.parameters()).device
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        _, loss_dict_no_ema = model.shared_step(batch)
        z, c, x_gt, xrec, xc = model.get_input(
            batch, model.first_stage_key, return_first_stage_outputs=True, force_c_encode=True, return_original_cond=True
        )
        relative_RT4 = batch["relative_RT4"]
        metrics = compute_evaluation_metrics(xrec, x_gt, relative_RT4)
        all_metrics.append(metrics)

# 4. Aggregate and plot
def collect_metric(metric_name):
    return np.array([m[metric_name] for m in all_metrics])

output_dir = "/txining/zero123/zero123/validation_results"
os.makedirs(output_dir, exist_ok=True)

for key in all_metrics[0].keys():
    values = collect_metric(key)
    print(f"{key}: mean={values.mean():.4f}, std={values.std():.4f}")
    plt.plot(values, label=key)
    # Save each metric as a .npy file
    # np.save(os.path.join(output_dir, f"{key}_values.npy"), values)

plt.legend()
plt.title("Validation Metrics")
plt.xlabel("Batch")
plt.ylabel("Metric Value")
plot_path = os.path.join(output_dir, "validation_metrics.png")
plt.savefig(plot_path)
plt.close()
print(f"Saved metrics and plot to {output_dir}")