#!/usr/bin/env python3
"""Train the two-branch mark classifier and export it to ONNX."""

import argparse
import json
import pathlib

try:
    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import functional as TF
except ImportError as error:
    raise SystemExit(
        "Training dependencies are missing. Install tools/requirements-mark-training.txt"
    ) from error


class MarkDataset(Dataset):
    def __init__(self, records, classes):
        self.records = records
        self.class_to_index = {name: index for index, name in enumerate(classes)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        mask = TF.to_tensor(Image.open(record["maskPath"]).convert("L"))
        context = TF.to_tensor(Image.open(record["contextPath"]).convert("L"))
        geometry_record = record["geometry"]
        geometry = torch.tensor(
            [
                geometry_record["width"],
                geometry_record["height"],
                geometry_record["area"],
                geometry_record["width"]
                / max(1.0, geometry_record["height"]),
                geometry_record["area"]
                / max(
                    1.0,
                    geometry_record["width"] * geometry_record["height"],
                ),
            ],
            dtype=torch.float32,
        )
        geometry = torch.log1p(geometry)
        return (
            mask,
            context,
            geometry,
            self.class_to_index[record["visualClass"]],
        )


class ImageEncoder(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, output_features),
            nn.ReLU(),
        )

    def forward(self, image):
        return self.network(image)


class MarkClassifier(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.mask_encoder = ImageEncoder(96)
        self.context_encoder = ImageEncoder(96)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(224, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, class_count),
        )

    def forward(self, candidate, context, geometry):
        features = torch.cat(
            [
                self.mask_encoder(candidate),
                self.context_encoder(context),
                self.geometry_encoder(geometry),
            ],
            dim=1,
        )
        return self.classifier(features)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for candidate, context, geometry, labels in loader:
            logits = model(
                candidate.to(device), context.to(device), geometry.to(device)
            )
            correct += (logits.argmax(1).cpu() == labels).sum().item()
            total += labels.numel()
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.annotations.read_text().splitlines()
        if line.strip()
    ]
    records = [
        record
        for record in records
        if record["reviewed"]
        and record["visualClass"] is not None
        and record.get("expectedText")
    ]
    classes = sorted({record["visualClass"] for record in records})
    if len(classes) < 2:
        raise SystemExit("At least two reviewed visual classes are required")

    datasets = {
        split: MarkDataset(
            [record for record in records if record["split"] == split], classes
        )
        for split in ("train", "validation", "test")
    }
    if not datasets["train"]:
        raise SystemExit("No reviewed training records")
    loaders = {
        split: DataLoader(
            dataset, batch_size=64, shuffle=split == "train", num_workers=0
        )
        for split, dataset in datasets.items()
    }

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MarkClassifier(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_validation = -1.0
    for _ in range(args.epochs):
        model.train()
        for candidate, context, geometry, labels in loaders["train"]:
            optimizer.zero_grad()
            loss = criterion(
                model(
                    candidate.to(device),
                    context.to(device),
                    geometry.to(device),
                ),
                labels.to(device),
            )
            loss.backward()
            optimizer.step()
        validation = evaluate(model, loaders["validation"], device)
        if validation > best_validation:
            best_validation = validation
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    model.cpu().eval()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "mark-classifier.pt")
    candidate = torch.zeros(1, 1, 96, 96)
    context = torch.zeros(1, 1, 128, 192)
    geometry = torch.zeros(1, 5)
    torch.onnx.export(
        model,
        (candidate, context, geometry),
        args.out / "mark-classifier.onnx",
        input_names=["candidate", "context", "geometry"],
        output_names=["logits"],
        dynamic_axes={
            "candidate": {0: "batch"},
            "context": {0: "batch"},
            "geometry": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    metrics = {
        "classes": classes,
        "validationAccuracy": best_validation,
        "testAccuracy": evaluate(model, loaders["test"], torch.device("cpu")),
        "trainingSamples": len(datasets["train"]),
        "validationSamples": len(datasets["validation"]),
        "testSamples": len(datasets["test"]),
    }
    (args.out / "mark-classifier.json").write_text(
        json.dumps(metrics, indent=2)
    )


if __name__ == "__main__":
    main()
