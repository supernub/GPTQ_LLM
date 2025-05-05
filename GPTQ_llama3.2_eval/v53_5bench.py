import os, sys, json, subprocess
from pathlib import Path
from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL

# ───────────── GPTQModel  ─────────────
GPTQ_TASKS = [
    "arc_challenge",   # ARC
    "hellaswag",       # HellaSwag
    "mmlu",            # MMLU
]

# lm‑eval CLI 跑剩余 2 个
CLI_TASKS = ["piqa", "winogrande"]


def run_gptq_tasks(model_id: str, out_prefix: str) -> None:
    """ARC / HellaSwag / MMLU """
    print(f"▶️  GPTQModel.eval → {', '.join(GPTQ_TASKS)}")
    GPTQModel.eval(
        model_id,
        framework=EVAL.LM_EVAL,
        tasks=GPTQ_TASKS,
        output_file=f"{out_prefix}_gptq.json",
    )
    print(f"✓ save result → {out_prefix}_gptq.json\n")


def run_cli_tasks(model_id: str, out_prefix: str) -> None:
    """调用 lm_eval CLI 跑 PIQA & Winogrande"""
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_id},dtype=bfloat16",
        "--tasks",
        ",".join(CLI_TASKS),
        "--device",
        "cuda",
        "--output_path",
        f"{out_prefix}_cli.json",
    ]
    print(f"▶️  lm_eval CLI → {' '.join(CLI_TASKS)}")
    subprocess.run(cmd, check=True)
    print(f"✓ save result → {out_prefix}_cli.json\n")


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python run_5bench.py /path/")
        sys.exit(1)

    model_id = sys.argv[1]

    out_prefix = Path(model_id).name.replace("/", "_")

    run_gptq_tasks(model_id, out_prefix)
    run_cli_tasks(model_id, out_prefix)

    print("🎉 全部 5 个基准评测完成！")


if __name__ == "__main__":
    main()
