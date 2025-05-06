import json, subprocess, pathlib, sys, os
from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL

MODEL_PATH = "Llama-3.2-1B-Instruct-gptqmodel-v53-4bit"
OUT_PREFIX = "llama32_1b_gptq_v53_4bit"

def run_gptqmodel():
    tasks = [
        EVAL.LM_EVAL.ARC_CHALLENGE,
        EVAL.LM_EVAL.MMLU,
        EVAL.LM_EVAL.HELLASWAG,
    ]
    out_file = f"{OUT_PREFIX}_arc_mmlu_hellaswag.json"
    GPTQModel.eval(
        model_id=MODEL_PATH,
        framework=EVAL.LM_EVAL,
        tasks=tasks,
        output_file=out_file,
    )
    return out_file

def run_cli(tasks_str, out_file):
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        f"--model_args=pretrained={MODEL_PATH},dtype=bfloat16",
        "--tasks", tasks_str,
        "--device", "cuda",
        f"--output_path={out_file}",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_file

def merge_results(files, merged_path):
    merged = {}
    for f in files:
        with open(f, "r") as fp:
            merged.update(json.load(fp))
    with open(merged_path, "w") as fp:
        json.dump(merged, fp, indent=2)
    print("➜  Summary saved to", merged_path)

if __name__ == "__main__":
    pathlib.Path("./results").mkdir(exist_ok=True)
    os.chdir("./results")

    f1 = run_gptqmodel()
    f2 = run_cli("piqa,winogrande",  f"{OUT_PREFIX}_piqa_winogrande.json")

    merge_results([f1, f2], f"{OUT_PREFIX}_v53_5bench.json")
