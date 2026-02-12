import sys, os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse
import datetime

# --------------------
# Fix sys.path so Python can find models/
# --------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.model import AgeGenderModel


# --------------------
# Dataset
# --------------------
class AgeGenderDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['img_name'])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Age
        age = torch.tensor(row['age'], dtype=torch.float32)

        # Gender robust mapping
        gender_str = str(row['gender']).strip().lower()
        if gender_str in ['m', 'male', '1']:
            gender = 0
        elif gender_str in ['f', 'female', '0']:
            gender = 1
        else:
            raise ValueError(f"Unknown gender label: {row['gender']}")
        gender = torch.tensor(gender, dtype=torch.long)

        return img, age, gender


# --------------------
# Evaluation function
# --------------------
def evaluate(model, dataloader, device, verbose=False):
    model.eval()
    mae_age = 0.0
    correct_gender = 0
    total = 0

    criterion_age = nn.L1Loss()  # MAE for age

    with torch.no_grad():
        for imgs, ages, genders in tqdm(dataloader, desc="Evaluating", unit="batch"):
            imgs, ages, genders = imgs.to(device), ages.to(device), genders.to(device)

            age_preds, gender_preds = model(imgs)

            # Age MAE
            mae_age += criterion_age(age_preds.squeeze(), ages).item() * imgs.size(0)

            # Gender accuracy
            predicted_genders = torch.argmax(gender_preds, dim=1)
            correct_gender += (predicted_genders == genders).sum().item()

            total += imgs.size(0)

            if verbose:
                print(f"Batch {total}: MAE={mae_age/total:.2f}, GenderAcc={(correct_gender/total)*100:.2f}%")

    avg_mae_age = mae_age / total
    gender_acc = correct_gender / total

    return avg_mae_age, gender_acc


# --------------------
# Main
# --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Age-Gender Model")
    parser.add_argument("--csv", type=str, default="data/val.csv", help="Validation CSV file")
    parser.add_argument("--img_dir", type=str, default="data/utk_val/val", help="Validation image directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/age_gender_model.pth", help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--save_log", type=bool, default=True, help="Save results to file")
    parser.add_argument("--verbose", action="store_true", help="Print per-batch metrics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------
    # Transforms
    # --------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # must match training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --------------------
    # Dataset & DataLoader
    # --------------------
    val_dataset = AgeGenderDataset(
        csv_file=args.csv,
        img_dir=args.img_dir,
        transform=transform
    )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --------------------
    # Load model
    # --------------------
    model = AgeGenderModel()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    # --------------------
    # Run evaluation
    # --------------------
    mae_age, gender_acc = evaluate(model, val_loader, device, verbose=args.verbose)

    print(f"\nValidation Results:")
    print(f"  Age MAE      : {mae_age:.2f}")
    print(f"  Gender Acc   : {gender_acc * 100:.2f}%")

    # --------------------
    # Save log
    # --------------------
    if args.save_log:
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/eval_{timestamp}.txt"
        with open(log_file, "w") as f:
            f.write(f"Checkpoint : {args.checkpoint}\n")
            f.write(f"CSV File   : {args.csv}\n")
            f.write(f"Images Dir : {args.img_dir}\n")
            f.write(f"Batch Size : {args.batch_size}\n\n")
            f.write(f"Validation Results:\n")
            f.write(f"  Age MAE      : {mae_age:.2f}\n")
            f.write(f"  Gender Acc   : {gender_acc * 100:.2f}%\n")
        print(f"\n✅ Results saved to {log_file}")
