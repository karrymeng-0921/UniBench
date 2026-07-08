import os
import torch
import numpy as np
import pandas as pd
from scipy import linalg
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_fid.inception import InceptionV3


class RecursiveImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.PNG', '.JPG'}

        for root, _, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file)[1] in valid_extensions:
                    self.image_paths.append(os.path.join(root, file))

        if len(self.image_paths) == 0:
            print(f"Warning: No valid images found in {root_dir}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def get_activation_statistics_recursive(dir_path, model, batch_size=50, device="cuda:0"):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    dataset = RecursiveImageDataset(dir_path, transform=transform)
    if len(dataset) == 0:
        return None, None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    act_list = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch = batch * 2.0 - 1.0
            pred = model(batch)[0]

            if pred.size(2) == 1 and pred.size(3) == 1:
                pred = pred.squeeze(3).squeeze(2)

            act_list.append(pred.cpu().numpy())

    act = np.concatenate(act_list, axis=0)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)

    return mu, sigma


def calculate_safe_fid_value(m1, s1, m2, s2, eps=1e-6):
    ssdiff = np.sum((m1 - m2) ** 2.0)

    offset = np.eye(s1.shape[0]) * eps
    s1_stable = s1 + offset
    s2_stable = s2 + offset

    covprod = s1_stable.dot(s2_stable)
    covmean, _ = linalg.sqrtm(covprod, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_value = ssdiff + np.trace(s1 + s2 - 2.0 * covmean)
    return float(fid_value)


def main_debug_fid():
    original_dir = "xxx"
    baselines_root = "xxx"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    print("Extracting statistics for the original images...")
    m_orig, s_orig = get_activation_statistics_recursive(
        original_dir,
        model,
        batch_size=50,
        device=device
    )

    if m_orig is None:
        print("Failed to extract statistics from the original image directory.")
        return

    print("Original statistics extracted successfully.")

    baseline_folders = sorted([
        f for f in os.listdir(baselines_root)
        if os.path.isdir(os.path.join(baselines_root, f))
    ])

    results = []

    for folder in baseline_folders:
        baseline_dir = os.path.join(baselines_root, folder)
        print(f"Processing: {folder}")

        m_base, s_base = get_activation_statistics_recursive(
            baseline_dir,
            model,
            batch_size=50,
            device=device
        )

        if m_base is None:
            print(f"Skipping {folder}: no valid images found.")
            results.append({
                "Baseline_Model": folder,
                "FID": float("nan")
            })
            continue

        fid = calculate_safe_fid_value(m_orig, s_orig, m_base, s_base)

        print(f"FID = {fid:.4f}")

        results.append({
            "Baseline_Model": folder,
            "FID": fid
        })

    df = pd.DataFrame(results)

    print("\nFID Evaluation Results:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main_debug_fid()
