
"""
Run 5 benchmarks (ARC, MMLU, HellaSwag, PIQA, Winogrande) for a local GPTQ model.

Prereqs:
  pip install gptqmodel==2.2.0 lm-eval==0.4.2 "datasets<3" evaluate==0.4.1
Folder layout:
  ├─ v53_5bench.py      (this file)
  └─ Llama-3.2-1B-Instruct-gptq-v53-4bit/
       ├─ config.json
       ├─ model.safetensors
       └─ ...

Run:
  python v53_5bench.py
"""

import json, subprocess, pathlib, sys, os
from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL

# ────────────────────────────────  配置  ────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / "Llama-3.2-1B-Instruct-gptq-v53-4bit").as_posix()

OUT_PREFIX = "llama32_1b_gptq_v53_4bit"
RESULT_DIR = BASE_DIR / "result"               
RESULT_DIR.mkdir(exist_ok=True)

# ────────────────────────────────  Helper  ────────────────────────────────
def run_gptq_three():
    """ARC, MMLU, HellaSwag — 由 GPTQModel.eval() 直接评测"""
    tasks = [
        EVAL.LM_EVAL.ARC_CHALLENGE,
        EVAL.LM_EVAL.MMLU,
        EVAL.LM_EVAL.HELLASWAG,
    ]
    out_file = RESULT_DIR / f"{OUT_PREFIX}_arc_mmlu_hellaswag.json"
    GPTQModel.eval(
        model_id=MODEL_PATH,
        framework=EVAL.LM_EVAL,
        tasks=tasks,
        output_file=out_file.as_posix(),
    )
    return out_file

def run_cli_piqa_wino():
    """PIQA & Winogrande — 用 lm‑eval‑harness CLI"""
    out_file = RESULT_DIR / f"{OUT_PREFIX}_piqa_winogrande.json"
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        f"--model_args=pretrained={MODEL_PATH},dtype=bfloat16",
        "--tasks", "piqa,winogrande",
        "--device", "cuda",
        f"--output_path={out_file.as_posix()}",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_file

def merge_json(files, merged_path):
    merged = {}
    for fp in files:
        with open(fp, "r") as f:
            merged.update(json.load(f))
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print("✓ Summary saved to", merged_path)

# ────────────────────────────────  Main  ────────────────────────────────
if __name__ == "__main__":
    os.chdir(RESULT_DIR)                       # 切到 ./result

    f_arc_mmlu_hsw = run_gptq_three()
    f_piqa_wino    = run_cli_piqa_wino()

    merge_json(
        [f_arc_mmlu_hsw, f_piqa_wino],
        f"{OUT_PREFIX}_5bench.json"
    )
