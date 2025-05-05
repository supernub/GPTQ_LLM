from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL

model_id = "Llama-3.2-1B-Instruct-gptqmodel-v53-4bit"

TASKS = [
    "arc_challenge",   # ARC
    "mmlu",            # MMLU
    "hellaswag",       # HellaSwag
    "piqa",            # PIQA
    "winogrande",      # Winogrande
]

lm_eval_results = GPTQModel.eval(
    model_id,
    framework=EVAL.LM_EVAL,
    tasks=TASKS,
    output_file="llama32_1b_4bit_v53_5bench.json",
)
print(lm_eval_results)
