from gptqmodel import GPTQModel, QuantizeConfig

model_id = "meta-llama/Llama-3.2-1B"
quant_path = "Llama-3.2-1B-gptq-v53-4bit"

with open("4o_v53.txt", "r", encoding="utf-8") as f:
    calibration_dataset = [line.strip() for line in f if line.strip()]

quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)

model.quantize(calibration_dataset, batch_size=2)

model.save(quant_path)

model = GPTQModel.load(quant_path)
result = model.generate("Uncovering deep insights begins with")[0]
print(model.tokenizer.decode(result))
