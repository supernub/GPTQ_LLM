from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL

TASKS = [
    EVAL.LM_EVAL.ARC_CHALLENGE,
    EVAL.LM_EVAL.MMLU,
    EVAL.LM_EVAL.HELLASWAG,
    EVAL.LM_EVAL.PIQA,
    EVAL.LM_EVAL.WINOGRANDE,
]

lm_eval_results = GPTQModel.eval(
    "Llama-3.2-1B-Instruct-gptqmodel-v53-4bit",
    framework=EVAL.LM_EVAL,
    tasks=TASKS,
    output_file="llama32_1b_4bit_lm_eval.json",
)
