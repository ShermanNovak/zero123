from ldm.modules.evaluate.evaluate_perceptualsim import PNet, perceptual_sim, ssim_metric, psnr
from ldm.modules.evaluate.consistency import compute_geometric_consistency_approx_F, compute_geometric_consistency_with_K, compute_geometric_consistency_with_latent, compute_geometric_consistency_with_translation
import torch
import numpy as np

def compute_evaluation_metrics(batch_img_rec, batch_img_gt, batch_relative_RT, calc_latent_sim=False, calc_trans_sim=False):
    """
    Compute evaluation metrics for a batch of images.
    Args:
        batch_img_rec: Tensor of reconstructed images, shape (B, C, H, W)
        batch_img_gt: Tensor of ground truth images, shape (B, C, H, W)
        batch_relative_RT: List or tensor of relative RTs, length B
    Returns:
        dict with average and std of metrics over the batch
    """

    vgg16 = PNet().to("cuda")
    vgg16.eval()
    vgg16.cuda()

    lpips_scores = []
    ssim_scores = []
    psnr_scores = []

    B = batch_img_rec.shape[0]
    for i in range(B):
        img_rec = batch_img_rec[i].unsqueeze(0).cuda()
        img_gt = batch_img_gt[i].unsqueeze(0).cuda()
        relative_RT = batch_relative_RT[i]

        # Perceptual similarity
        perc_sim = perceptual_sim(img_rec, img_gt, vgg16).item()
        lpips_scores.append(perc_sim)

        # SSIM
        ssim_score = ssim_metric(img_rec, img_gt).item()
        ssim_scores.append(ssim_score)

        # PSNR
        psnr_score = psnr(img_rec, img_gt).item()
        psnr_scores.append(psnr_score)

    # Geometric consistency
    gc_K = compute_geometric_consistency_with_K(batch_img_rec, batch_img_gt, batch_relative_RT)
    gc_F = compute_geometric_consistency_approx_F(batch_img_rec, batch_img_gt)

    def avg_std(x):
        return float(np.nanmean(x)), float(np.nanstd(x))

    avg_percsim, std_percsim = avg_std(lpips_scores)
    avg_ssim, std_ssim = avg_std(ssim_scores)
    avg_psnr, std_psnr = avg_std(psnr_scores)

    metrics = {
        "LPIPS/avg": avg_percsim,
        "LPIPS/std": std_percsim,
        "PSNR/avg": avg_psnr,
        "PSNR/std": std_psnr,
        "SSIM/avg": avg_ssim,
        "SSIM/std": std_ssim,
        "gc_K/avg": gc_K,
        "gc_F/avg": gc_F,
    }

    if calc_latent_sim:
        gc_L = compute_geometric_consistency_with_latent(batch_img_rec, batch_img_gt, batch_relative_RT)
        metrics["gc_L/avg"] = gc_L
    if calc_trans_sim:
        gc_T = compute_geometric_consistency_with_translation(batch_img_rec, batch_img_gt, batch_relative_RT)
        metrics["gc_T/avg"] = gc_T

    return metrics