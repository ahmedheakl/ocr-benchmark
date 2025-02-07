import datasets
import os
from tqdm import tqdm
import json
from qwen2vl_model import Qwen2VLOCR

DEFAULT_PROMPT = "Extract the text in the image. Give me the final text, nothing else."
RESULTS_DIR = "/home/omkar/ocr-eval/results"
MAX_TOKENS = 2000
NUM_WORKERS = 4
os.makedirs(RESULTS_DIR, exist_ok=True)

ds_ids = [
    "ahmedheakl/arocrbench_patsocr", # answer
    "ahmedheakl/arocrbench_historicalbooks", # answer
    "ahmedheakl/arocrbench_khattparagraph", # answer
    "ahmedheakl/arocrbench_synthesizear",
    "ahmedheakl/arocrbench_historyar",
    "ahmedheakl/arocrbench_adab",
    "ahmedheakl/arocrbench_muharaf",
    "ahmedheakl/arocrbench_onlinekhatt",
    "ahmedheakl/arocrbench_khatt",
    "ahmedheakl/arocrbench_isippt",
    "ahmedheakl/arocrbench_arabicocr",
    "ahmedheakl/arocrbench_hindawi",
    "ahmedheakl/arocrbench_evarest",

]

model_name = "qwen2vl7b"

for ds_id in tqdm(ds_ids):
    ds_name = ds_id.split("_")[-1]
    print(f"Evaluating {ds_name} ...")
    output_path = f"{RESULTS_DIR}/{model_name}_{ds_name}.json"
    # if os.path.exists(output_path): continue
    ds = datasets.load_dataset(ds_id, split="train", num_proc=NUM_WORKERS)
    data = []
    model = Qwen2VLOCR(MAX_TOKENS)
    answer_name = "answer" if ds_id in ds_ids[:3] else "text"
    for idx, sample in tqdm(enumerate(ds), total=len(ds), desc=f"Evaluating {ds_name}"):
        img = sample['image']
        try:
            pred = model(DEFAULT_PROMPT, img)
        except Exception as e:
            print(f"Skipping {idx} for {e}")
            pred = ""
        gt = sample[answer_name]
        data.append({"idx": idx, "gt": gt, "pred": pred})


    with open(output_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
