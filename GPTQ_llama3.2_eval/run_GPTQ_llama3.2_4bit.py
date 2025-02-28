from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
huggingface-cli login

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
).select(range(1024))["text"]  # 仅选取前1024


quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)

model.quantize(calibration_dataset, batch_size=2)

model.save(quant_path)

# test
model = GPTQModel.load(quant_path)
prompt_text = "Uncovering deep insights begins with"
result_tokens = model.generate(prompt_text)[0]
result_str = model.tokenizer.decode(result_tokens)

print("===== Prompt =====")
print(prompt_text)
print("===== Generated Output =====")
print(result_str)