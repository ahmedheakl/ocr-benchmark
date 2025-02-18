import datasets
import os
from tqdm import tqdm
import json
from tesseract_model import TesseractOCR

RESULTS_DIR = "results"
NUM_WORKERS = 2
os.makedirs(RESULTS_DIR, exist_ok=True)

model_name = "tesseract"
output_path = f"{RESULTS_DIR}/{model_name}.json"
ds = datasets.load_dataset("ahmedheakl/arocrbench_our_lines_v3", split="train")
data = []

ocr_model = TesseractOCR()

for idx, sample in tqdm(enumerate(ds), total=len(ds), desc=f"Evaluating lines"):
  img = sample['image']
  try:
      pred = ocr_model(img)
  except Exception as e:
      print(f"Skipping {idx} for {e}")
      pred = ""
  gt = sample['data']
  data.append({"idx": idx, "gt": gt, "pred": pred})

with open(output_path, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)