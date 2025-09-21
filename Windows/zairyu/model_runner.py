import os
import sys
from types import SimpleNamespace

# 更彻底地阻止xformers导入
class MockXformers:
    def __getattr__(self, name):
        raise ImportError(f"xformers.{name} is disabled")

class MockXformersOps:
    def memory_efficient_attention(self, q, k, v, p=0.0):
        # 使用PyTorch原生的scaled dot product attention
        import torch.nn.functional as F
        
        # 重塑张量以适应标准attention格式
        B, N, H, D = q.shape
        q = q.transpose(1, 2)  # (B, H, N, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 使用PyTorch原生attention
        with torch.nn.attention.sdpa_kernel(
            backends=[torch.nn.attention.SDPBackend.MATH]
        ):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        
        # 重塑回原来的格式
        out = out.transpose(1, 2)  # (B, N, H, D)
        return out

sys.modules['xformers'] = MockXformers()
sys.modules['xformers.ops'] = MockXformersOps()

# 同时设置环境变量强制禁用
os.environ['XFORMERS_DISABLED'] = '1'


# 跳过不兼容的加速和节省显存功能A
from types import SimpleNamespace

try:
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
except ImportError:
    sys.modules["transformers.models.llama.modeling_llama"].LlamaFlashAttention2 = None

# 正统环境
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.utils.io import load_pil_images


from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

class OCRModel:
    def __init__(self, model_path="deepseek-ai/deepseek-vl2-tiny",use_cpu=False):
        # 处理器
        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        if use_cpu:
            self.device = torch.device("cpu")
            dtype = torch.float32
            device_map = None
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("未检测到 GPU，请使用 use_cpu=True")
            self.device = torch.device("cuda")
            dtype = torch.float32  
            device_map = "auto"

        self.model = DeepseekVLV2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True
        ).to(self.device).eval()

        self.model.config.use_cache = False
    def predict(self, image_path):
        conversation = [
            {
                "role": "<|User|>",
                "content": """<image>\n
                言語設定：日本語！。
                以下の項目だけ答えてください：
                - 国籍（例：ベトナム、韓国、中国など）
                - 氏名（氏名は原文の通り、たとえ英語でもカタカナに変換しないでください。）
                - 性別（男 または 女）
                - 生年月日
                - 住居地（都道府県から）
                - 在留資格（例：留学、家族滞在など）
                - 在留カード番号（右上の番号は在留カード番号です）
                - 在留期間(PERIOD OF STAY):X年X月 (XXXX年XX月XX日)
                それ以外の説明や解説は不要です。""",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # 1. 处理图像
        pil_images = load_pil_images(conversation)
        inputs = self.processor(conversations=conversation, images=pil_images, force_batchify=True)
        inputs = inputs.to(self.device)

        # 2. 强制 float32 处理图像特征
        inputs_dict = dict(inputs)
        for k, v in inputs_dict.items():
            if isinstance(v, torch.Tensor) and k in ["pixel_values", "images"]:
                inputs_dict[k] = v.to(torch.float32)

        # 3. 转换成 embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**inputs_dict)

        # 4. 推理生成
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs_dict.get("attention_mask"),
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=500,
            do_sample=False,
            use_cache=True,
        )

        # 5. 解码输出
        return self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
