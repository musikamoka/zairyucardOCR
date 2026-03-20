import torch
import os
from transformers import AutoModel, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = r"D:\desk\26OCR\DeepSeek-OCR-2"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='eager',
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

image_file = r"D:\desk\26OCR\zairyu-card-1.jpg"

# 第一次：Free OCR 提取所有文字（包括番号）
res1 = model.infer(
    tokenizer,
    prompt="<image>\nFree OCR. ",
    image_file=image_file,
    output_path=r"D:\desk\26OCR\output\free",
    base_size=1024,
    image_size=1024,
    crop_mode=True,
    save_results=True
)

# 第二次：Markdown 模式提取结构
res2 = model.infer(
    tokenizer,
    prompt="<image>\n<|grounding|>Convert the document to markdown. ",
    image_file=image_file,
    output_path=r"D:\desk\26OCR\output\markdown",
    base_size=1024,
    image_size=1024,
    crop_mode=True,
    save_results=True
)

print("=== Free OCR ===")
print(res1)
print("=== Markdown ===")
print(res2)