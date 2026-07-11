import os
import json
import torch
import argparse
import warnings
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel
from model_Multitask3 import (
    AdvancedClassifierHead_CLIP,
    MLPHead,
    MultiHeadCLIPClassifier,
   StyleAttentionMLPHead
)
from util import fix_seed
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from PIL import Image
import psutil
import random

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Task metadata ----------
task_type_dict = {
    "object": "multiclass",
    "style": "multiclass",
    "nsfw": "multiclass", 
}

# ---------- Utility functions ----------
def estimate_batch_size(dataset_path, target_mem_ratio=0.8, img_size=(224, 224), channels=3, dtype_bytes=4):
    mem = psutil.virtual_memory()
    avail_bytes = mem.available
    safe_bytes = avail_bytes * target_mem_ratio
    img_bytes = img_size[0] * img_size[1] * channels * dtype_bytes
    batch_size = int(safe_bytes // img_bytes)
    return max(1, batch_size)

def save_dataset_to_pth(dataset_path, save_path, batch_size=512, img_size=(224, 224)):
    os.makedirs(save_path, exist_ok=True)
    if any(f.endswith(".pth") for f in os.listdir(save_path)):
        print(f"[INFO] Dataset already saved in {save_path}, skipping...")
        return

    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    ds = datasets.ImageFolder(dataset_path)
    all_tensors, all_labels, all_domains = [], [], []
    batch_idx = 0

    for i, (img_path, _) in enumerate(tqdm(ds.samples, desc=f"Saving {save_path}")):
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img)
        all_tensors.append(tensor)
        label = os.path.basename(os.path.dirname(img_path))
        all_labels.append(label)
        domain = "sdgen" if "sd" in os.path.basename(img_path).lower() else "real"
        all_domains.append(domain)

        if (i + 1) % batch_size == 0 or i + 1 == len(ds):
            batch_file = os.path.join(save_path, f"batch{batch_idx}.pth")
            torch.save({"data": torch.stack(all_tensors), "labels": all_labels, "domains": all_domains}, batch_file)
            batch_idx += 1
            all_tensors, all_labels, all_domains = [], [], []

class MemoryDataset(Dataset):
    def __init__(self, pth_dir, augment=False, task_name=None):
        self.pth_files = sorted([os.path.join(pth_dir, f) for f in os.listdir(pth_dir) if f.endswith(".pth")])
        self.augment = augment
        self.task_name = task_name
        self.data_dicts = [torch.load(f, map_location="cpu") for f in self.pth_files]

        all_labels = [lbl for d in self.data_dicts for lbl in d["labels"]]
        self.label2idx = {l: i for i, l in enumerate(sorted(set(all_labels)))}

        if augment and task_name == "style":
            self.real_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(5),
                transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
            ])
            self.gen_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        else:
            self.real_transform = self.gen_transform = None

        self.flat_data = [(d_idx, i) for d_idx, d in enumerate(self.data_dicts) for i in range(len(d["labels"]))]

    def __len__(self):
        return len(self.flat_data)

    def __getitem__(self, idx):
        d_idx, i = self.flat_data[idx]
        data_dict = self.data_dicts[d_idx]
        image = data_dict["data"][i]
        label_str = data_dict["labels"][i]
        label = self.label2idx[label_str]
        domain = data_dict["domains"][i]

        if self.augment and self.task_name == "style":
            if domain == "real" and self.real_transform:
                image = self.real_transform(transforms.ToPILImage()(image))
                image = transforms.ToTensor()(image)
            elif domain == "sdgen" and self.gen_transform:
                image = self.gen_transform(transforms.ToPILImage()(image))
                image = transforms.ToTensor()(image)

        image = image * 2.0 - 1.0
        return {"pixel_values": image, "labels": torch.tensor(label, dtype=torch.long), "label_str": label_str}

def make_loader_from_pth(pth_dir, batch_size, augment=False, task_name=None):
    dataset = MemoryDataset(pth_dir, augment=augment, task_name=task_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=augment, num_workers=0)

def load_clip():
    clip_model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    return clip_model, processor

# ---------- PGD adversarial attack ----------
def pgd_attack(model, images, labels, task, eps=8/255, alpha=2/255, iters=3):
    model.eval()
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    delta = torch.zeros_like(images).uniform_(-eps, eps).to(device)
    delta.requires_grad = True
    for _ in range(iters):
        outputs = model(images + delta, task_name=task, labels=labels, detach_clip=False)
        loss = outputs["loss"]
        loss.backward()
        grad = delta.grad.detach()
        delta.data = (delta + alpha * torch.sign(grad)).clamp(-eps, eps)
        delta.grad.zero_()
    return (images + delta).clamp(-1, 1)

# ---------- Evaluation ----------
def evaluate(model, loader, task):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values, task_name=task, labels=labels, detach_clip=True)
            logits = outputs["logits"]
            if task_type_dict[task] == "multiclass":
                preds = torch.argmax(logits, dim=1)
            else:
                preds = (torch.sigmoid(logits.squeeze()) > 0.5).long()
            total_correct += (preds == labels).sum().item()
            total_loss += outputs["loss"].item() * labels.size(0)
            total_samples += labels.size(0)
    return total_loss / total_samples, total_correct / total_samples

# ---------- Plot metrics ----------
def plot_metrics(metrics, save_dir, task):
    plt.figure(figsize=(10, 4))
    for i, metric in enumerate(["loss", "acc"]):
        plt.subplot(1, 2, i + 1)
        for split in metrics:
            if metric in metrics[split]:
                plt.plot(metrics[split][metric], label=split, marker='o')
        plt.title(f"{task} {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.upper())
        plt.legend()
        plt.grid(True)
        if metric == "acc":
            plt.ylim(0, 1.05)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{task}_metrics.png"))
    plt.close()

# ---------- Main function ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, nargs="+", default=["object", "style", "nsfw"],
                        help="Tasks to train: choose from object, style, nsfw")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-dir", type=str, default="CLASSIFIED_IMAGES_DIR",
                        help="Root directory containing train/val subdirectories for each task")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to save training results")
    parser.add_argument("--batch-size-save", type=int, default=None)
    parser.add_argument("--adv-ratio", type=float, default=0.5)
    args = parser.parse_args()

    fix_seed(42)
    os.makedirs(args.results_dir, exist_ok=True)

    # ---------- Step 1: Save datasets ----------
    datasets_to_save = [
        ("object/train", "object_train"),
        ("object/val/Real", "object_val_real"),
        ("object/val/SD-Gen", "object_val_sdgen"),
        ("style/train", "style_train"),
        ("style/val/Real", "style_val_real"),
        ("style/val/SD-Gen", "style_val_sdgen"),
        ("nsfw/train", "nsfw_train"),
        ("nsfw/val/Real", "nsfw_val_real"),
        ("nsfw/val/SD-Gen", "nsfw_val_sdgen"),
    ]

    for src, dst in datasets_to_save:
        dataset_path = os.path.join(args.base_dir, src)
        save_path = os.path.join(dst)
        batch_size_save = args.batch_size_save or estimate_batch_size(dataset_path)
        save_dataset_to_pth(dataset_path, save_path, batch_size=batch_size_save)

    # ---------- Step 2: Task-wise training ----------
    for task in args.tasks:
        if task not in ["object", "style", "nsfw"]:
            print(f"[WARNING] Unknown task {task}, skipping...")
            continue

        print(f"\n==== Task: {task} ====")
        clip_model, processor = load_clip()

        head_dict = {
            "object": AdvancedClassifierHead_CLIP(input_dim=512, hidden_dim=512, num_classes=10),
            "style": StyleAttentionMLPHead(input_dim=512, hidden_dim=512, num_classes=20, dropout_rate=0.3),
            "nsfw": MLPHead(input_dim=512, hidden_dim=256, num_classes=7, dropout=0.3),
        }

        model = MultiHeadCLIPClassifier(clip_model, {task: head_dict[task]}, {task: task_type_dict[task]}).to(device)

        log_file_path = os.path.join(args.results_dir, f"{task}_epoch_logs.json")
        with open(log_file_path, "w") as f:
            json.dump([], f)

        # ---------- Phase 1: Full-model fine-tuning ----------
        print(f"[Phase 1] Full-model fine-tune with PGD")
        for param in model.clip_model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=5e-6, weight_decay=1e-3)

        train_loader = make_loader_from_pth(f"{task}_train",
                                            batch_size=args.batch_size, augment=True, task_name=task)

        for epoch in range(args.pretrain_epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                pv = batch["pixel_values"].to(device)
                lbl = batch["labels"].to(device)
                adv = pgd_attack(model, pv, lbl, task, alpha=0.53/255, iters=15)
                mixed = torch.cat([adv, pv], dim=0)
                mixed_lbl = torch.cat([lbl, lbl], dim=0)
                outputs = model(mixed, task_name=task, labels=mixed_lbl)
                outputs["loss"].backward()
                optimizer.step()

        # ---------- Phase 2: Freeze backbone, train head only ----------
        print(f"[Phase 2] Freeze CLIP backbone, train head only")
        for param in model.clip_model.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(model.head_dict[task].parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        metrics = {"train": {"loss": [], "acc": []},
                   "val_real": {"loss": [], "acc": []},
                   "val_sdgen": {"loss": [], "acc": []}}

        for epoch in range(args.epochs):
            model.train()
            total_loss, total_correct, total_samples = 0, 0, 0
            for batch in train_loader:
                optimizer.zero_grad()
                pv = batch["pixel_values"].to(device)
                lbl = batch["labels"].to(device)
                adv = pgd_attack(model, pv, lbl, task, alpha=0.53/255, iters=15)
                mixed = torch.cat([adv, pv], dim=0)
                mixed_lbl = torch.cat([lbl, lbl], dim=0)
                outputs = model(mixed, task_name=task, labels=mixed_lbl)
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=1)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * mixed_lbl.size(0)
                total_correct += (preds == mixed_lbl).sum().item()
                total_samples += mixed_lbl.size(0)
            avg_loss, avg_acc = total_loss / total_samples, total_correct / total_samples
            metrics["train"]["loss"].append(avg_loss)
            metrics["train"]["acc"].append(avg_acc)

            # ---------- Validation ----------
            for domain in ["Real", "SD-Gen"]:
                tag = f"val_{domain.lower().replace('-', '')}"
                val_loader = make_loader_from_pth(f"{task}_val_{domain.lower().replace('-', '')}",
                                                  batch_size=args.batch_size, augment=False, task_name=task)
                loss, acc = evaluate(model, val_loader, task)
                metrics[tag]["loss"].append(loss)
                metrics[tag]["acc"].append(acc)
            scheduler.step(metrics["val_real"]["loss"][-1])

            # ---------- Logging ----------
            epoch_record = {
                "task": task,
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_acc": avg_acc,
                "val_real_loss": metrics["val_real"]["loss"][-1],
                "val_real_acc": metrics["val_real"]["acc"][-1],
                "val_sdgen_loss": metrics["val_sdgen"]["loss"][-1],
                "val_sdgen_acc": metrics["val_sdgen"]["acc"][-1],
            }
            with open(log_file_path, "r+") as f:
                logs = json.load(f)
                logs.append(epoch_record)
                f.seek(0)
                json.dump(logs, f, indent=2)
                f.truncate()

        plot_metrics(metrics, args.results_dir, task)

        full_model_path = os.path.join(args.results_dir, f"{task}_full_model.pth")
        torch.save({
            "clip_backbone": model.clip_model.state_dict(),
            "head_dict": {task: model.head_dict[task].state_dict()},
            "task_type_dict": {task: task_type_dict[task]}
        }, full_model_path)
        print(f" Saved full model for {task} -> {full_model_path}")

    print("\nAll tasks complete. Each task now has its own isolated full model file.")

if __name__ == "__main__":
    main()
