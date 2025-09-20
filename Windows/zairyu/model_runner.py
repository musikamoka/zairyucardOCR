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
    def __init__(self, model_path="deepseek-ai/deepseek-vl2-tiny"):
        # 处理器
        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        # 模型（默认 CPU，可以切换到 CUDA）
        assert torch.cuda.is_available(), "需要可用的 CUDA/GPU"
        self.device = torch.device("cuda")
        
        # 设置精度 - 使用半精度以节省显存
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.model = DeepseekVLV2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  
            device_map="auto",          # 自动设备映射
            trust_remote_code=True
        ).eval()
        self.model.config.use_cache = False  # 禁用缓存避免版本不匹配
        
    def predict(self, image_path):
        conversation = [
            {
                "role": "<|User|>",
                "content": """<image>\n
                言語設定：日本語。  
                以下の項目だけ答えてください：
                - 国籍               
                - 氏名
                - 性別:(男|女) 
                - 生年月日  
                - 住居地 
                - 在留資格
                - 右上の番号は在留カード番号です:
                - 在留期間(PERIOD OF STAY):X年 (XXXX年XX月XX日)   
                それ以外の説明や解説は不要です。""",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
#需要加张量
        # 加载并处理图像
        pil_images = load_pil_images(conversation)

        # 准备输入
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(self.device)
        
        # 直接使用processor的generate方法
        # 替换整个生成部分
        with torch.no_grad():
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            
            # 直接前向传播
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask
            )
            
            # 获取logits并解码
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # 只取生成的新token部分（跳过输入部分）
            input_length = inputs_embeds.shape[1]
            generated_ids = predicted_ids[0, input_length:]
            
            return self.tokenizer.decode(generated_ids.cpu().tolist(), skip_special_tokens=True)