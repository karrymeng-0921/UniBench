import os
import json
import csv
import torch
import argparse
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
from model_Multitask3 import (
    AdvancedClassifierHead_CLIP,
    MLPHead,
    MultiHeadCLIPClassifier,
    Multi_MultiC_GramCluster_v2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Task heads definition (same as training)
HEADS = {
    "object_church": AdvancedClassifierHead_CLIP(input_dim=512, hidden_dim=512, num_classes=10),
    "style_vangogh": Multi_MultiC_GramCluster_v2(
        input_dim=512, hidden_dim=512, num_classes=20,
        dropout_rate=0.3, gram_reduce_dim=512, cluster_factor=3
    ),
    "nsfw": MLPHead(input_dim=512, hidden_dim=256, num_classes=7, dropout=0.3),
}

# Target classes (folder names used during training)
TARGET_LABEL = {
    "object_church": "2",  # church folder name
    "style_vangogh": "22",  # vangogh folder name
    "nsfw_nudenet": "4"      # Sexual folder name
}

# Dataset
class FlatFolderDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_path": os.path.basename(image_path),
        }

def make_loader(path, processor, batch_size=32, num_workers=4):
    dataset = FlatFolderDataset(path, processor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Load model and heads
def load_model_and_heads(weight_dir, task_alias):
    clip_model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name)

    alias_to_task = {
        "object_church": "object",
        "style_vangogh": "style",
        "nsfw_nudenet": "nsfw"
    }
    task_real = alias_to_task[task_alias]

    head_dict = {task_real: HEADS[task_alias]}
    task_type_dict = {task_real: "multiclass"}

    model = MultiHeadCLIPClassifier(clip_model, head_dict, task_type_dict).to(device)

    filename_map = {
        "object": "object_full_model.pth",
        "style": "style_full_model.pth",
        "nsfw": "nsfw_full_model.pth"
    }
    model_path = os.path.join(weight_dir, filename_map[task_real])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[✗] Missing model file: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model.clip_model.load_state_dict(checkpoint["clip_backbone"], strict=False)
    model.head_dict[task_real].load_state_dict(checkpoint["head_dict"][task_real])
    print(f"[✓] Loaded model and head for {task_real} from {model_path}")

    model.eval()
    return model, processor, task_real

# Multiclass evaluation
def evaluate_multiclass(model, loader, task_alias, task_real):
    # Build label2idx mapping used during training
    if task_alias == "style_vangogh":
        sorted_labels = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                         '2', '20', '22', '25', '3', '4', '5', '6', '7', '9']
        label2idx = {l: i for i, l in enumerate(sorted_labels)}
    elif task_alias == "object_church":
        object_dirs = [str(i) for i in range(10)]
        label2idx = {l: i for i, l in enumerate(sorted(object_dirs))}
    elif task_alias == "nsfw":
        nsfw_dirs = [str(i) for i in range(7)]
        label2idx = {l: i for i, l in enumerate(sorted(nsfw_dirs))}
    else:
        raise ValueError(f"Unknown task_alias: {task_alias}")

    target_idx = label2idx[TARGET_LABEL[task_alias]]

    all_preds = []
    all_paths = []
    target_count = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            outputs = model(pixel_values, task_name=task_real)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_paths.extend(batch["image_path"])

            target_count += (preds == target_idx).sum().item()
            total += preds.size(0)

    target_ratio = target_count / total if total > 0 else 0.0
    return target_count, total, target_ratio, list(zip(all_paths, all_preds))

# Main function
def main():
    parser = argparse.ArgumentParser("Unlearning Evaluation")
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Root directory containing images to evaluate")
    parser.add_argument("--weight-dir", type=str, required=True,
                        help="Directory containing trained classifier weights")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory to save evaluation results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tasks", nargs="+", default=list(HEADS.keys()),
                        help="Tasks to evaluate (e.g. object_church style_vangogh nsfw)")
    parser.add_argument("--methods", nargs="+", required=True,
                        help="Unlearning methods to evaluate (e.g. ESD FMN UCE)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    overall_log = {}
    detailed_list = []

    for task in args.tasks:
        if task not in HEADS:
            print(f"[!] Unknown task: {task}, skipping.")
            continue

        overall_log[task] = {}
        model, processor, task_real = load_model_and_heads(args.weight_dir, task)

        for method in args.methods:
            data_path = os.path.join(args.base_dir, task, "forget", method)
            if not os.path.exists(data_path):
                print(f"[!] Path not found: {data_path}, skipping.")
                continue

            loader = make_loader(data_path, processor, args.batch_size)
            target_count, total_count, ratio, preds_with_paths = evaluate_multiclass(
                model, loader, task, task_real
            )

            overall_log[task][method] = {
                "target_class_count": int(target_count),
                "total": int(total_count),
                "target_class_ratio": round(ratio, 4)
            }

            print(f"[{task} - {method}] "
                  f"Target class count: {target_count}/{total_count} ({ratio:.4f})")

            for path, pred in preds_with_paths:
                detailed_list.append({
                    "image_name": path,
                    "task": task,
                    "unlearning": method,
                    "prediction": int(pred)
                })

    # ===== Save JSON summary =====
    json_path = os.path.join(args.save_dir, "unlearn_eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall_log, f, indent=2, ensure_ascii=False)

    # ===== Save CSV (per-image) =====
    csv_path = os.path.join(args.save_dir, "unlearn_predictions.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "task", "unlearning", "prediction"])
        for record in detailed_list:
            writer.writerow([
                record["image_name"],
                record["task"],
                record["unlearning"],
                record["prediction"]
            ])

    print("\n[✓] Evaluation complete.")
    print(f"[✓] JSON saved to: {json_path}")
    print(f"[✓] CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
