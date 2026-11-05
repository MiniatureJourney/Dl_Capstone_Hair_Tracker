import os
import sys
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    TOKEN = input("Enter your Hugging Face write token (or set HF_TOKEN environment variable): ").strip()
REPO_ID = os.environ.get("HF_REPO_ID", "AlgomCorp/hair-segmentation-unet")

api = HfApi(token=TOKEN)

print("Target Repository:", REPO_ID)
print("Uploading best_model.pth (~88.9 MB)...")

# 1. Upload model weights
weights_path = r"C:\Users\Aliasgar\Desktop\DL Capstone\DL_Capstone_HairSegmentation\deployment\best_model.pth"
api.upload_file(
    path_or_fileobj=weights_path,
    path_in_repo="best_model.pth",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload U-Net best model weights (Dice: 95.19%)"
)
print("Uploaded best_model.pth!")

# 2. Upload architecture code
model_py_path = r"C:\Users\Aliasgar\Desktop\DL Capstone\DL_Capstone_HairSegmentation\deployment\model.py"
api.upload_file(
    path_or_fileobj=model_py_path,
    path_in_repo="model.py",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload U-Net PyTorch architecture implementation"
)
print("Uploaded model.py!")

# 3. Upload README.md (Model Card)
readme_path = r"C:\Users\Aliasgar\Desktop\DL Capstone\DL_Capstone_HairSegmentation\deployment\README.md"
api.upload_file(
    path_or_fileobj=readme_path,
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload Model Card, Evaluation Benchmarks & Documentation"
)
print("Uploaded README.md!")

# 4. Upload sample images
for fname in ["baseline_sample.png", "followup_sample.png"]:
    fpath = os.path.join(r"C:\Users\Aliasgar\Desktop\DL Capstone\DL_Capstone_HairSegmentation\deployment\examples", fname)
    if os.path.exists(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=f"examples/{fname}",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Upload sample image {fname}"
        )
        print(f"Uploaded examples/{fname}!")

print("\nSUCCESS: All files successfully published to Hugging Face Model Hub!")
print(f"URL: https://huggingface.co/{REPO_ID}")
