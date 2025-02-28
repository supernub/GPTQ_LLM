# pip install langchain transformers accelerate sentencepiece

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# 本地模型 deepseek-ai/DeepSeek-R1-Distill-Llama-8B
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,  # float16 / int8 
)

# text-generation
generate_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1024,
    temperature=0.1,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=generate_pipeline)

#  PDF 将其拆分成较小块，避免超出模型上下文限制
loader = PyPDFLoader("GPTQ.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(docs)

# 对分块内容提取量化方法的流程
prompt = PromptTemplate(
    input_variables=["text"], 
    template="阅读以下文本片段，找出并总结其中涉及的量化方法:\n{text}\n\n总结："
)
chain = LLMChain(llm=llm, prompt=prompt)

print("Running chain on the first chunk as an example:")
first_chunk_result = chain.run(text=chunks[0].page_content)
print("=== Summary for chunk #1 ===")
print(first_chunk_result)

# 若需要向量检索， HuggingFaceEmbeddings + FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

query = "论文中使用了哪些后训练量化方法？"
matched_docs = vectorstore.similarity_search(query, k=3)

print("\nTop 3 relevant chunks:")
for i, doc in enumerate(matched_docs):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content[:200] + "...")

# 把检索到的文本再次传给 LLM 进行问答
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="根据以下内容回答问题。\n内容：{context}\n问题：{question}\n\n回答："
)
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

for i, doc in enumerate(matched_docs):
    answer = qa_chain.run(context=doc.page_content, question=query)
    print(f"\nChunk {i+1} 回答：")
    print(answer)
