import os
import json
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.registry import get_dataset
from src.models.baseline import Baseline
from src.losses.cross_entropy import build_cross_entropy
from src.losses.triplet import BatchHardTripletLoss
from src.samplers.pk_sampler import PKSampler
from src.engine.trainer import train_one_epoch
from src.engine.evaluator import evaluate


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="market1501")
parser.add_argument("--model", type=str, default="resnet50")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--P", type=int, default=16)
parser.add_argument("--K", type=int, default=4)
parser.add_argument("--lr", type=float, default=3e-4)

args = parser.parse_args()

config = vars(args)


run_id = time.strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("runs", run_id)
os.makedirs(run_dir, exist_ok=True)

with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)



transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor()
])

DatasetClass = get_dataset(config["dataset"])

train_dataset = DatasetClass("data/clean/market1501", "train", transform)
query_dataset = DatasetClass("data/clean/market1501", "query", transform)
gallery_dataset = DatasetClass("data/clean/market1501", "gallery", transform)

sampler = PKSampler(train_dataset, P=config["P"], K=config["K"])

train_loader = DataLoader(train_dataset, batch_size=config["P"]*config["K"], sampler=sampler)
query_loader = DataLoader(query_dataset, batch_size=64, shuffle=False)
gallery_loader = DataLoader(gallery_dataset, batch_size=64, shuffle=False)


model = Baseline(num_classes=train_dataset.num_classes(), pretrained=True)

ce_loss = build_cross_entropy()
triplet_loss = BatchHardTripletLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])


for epoch in range(config["epochs"]):
    stats = train_one_epoch(model, train_loader, optimizer, ce_loss, triplet_loss)

    eval_stats = evaluate(model, query_loader, gallery_loader)

    log = {
        "epoch": epoch,
        "train_loss": stats,
        "eval": eval_stats
    }

    print(log)

    with open(os.path.join(run_dir, "log.txt"), "a") as f:
        f.write(json.dumps(log) + "\n")

print("Training complete.")