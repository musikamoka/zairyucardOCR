import torch
import os
from transformers import AutoModel, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = r"D:\desk\26OCR\DeepSeek-OCR-2" #权重位置

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    _attn_implementation="eager",
    use_safetensors=True
)

model = model.eval().cuda()

# ⚠️ 不要强制 bfloat16（很多卡不支持，会 silently 出问题）
# model = model.to(torch.bfloat16)

res = model.infer(
    tokenizer,
    prompt="<image>\n<|grounding|>Convert the document to markdown. ",
    image_file=r"D:\desk\26OCR\zairyu-card-1.jpg",   #文件位置
    output_path="./output",
    base_size=1024,
    image_size=768,
    crop_mode=True,
    save_results=True
)

print("RESULT:", res)