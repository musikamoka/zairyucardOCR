import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

class OCRModel:
    def __init__(self, model_path="deepseek-ai/deepseek-vl2-tiny"):
        self.processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.model = DeepseekVLV2ForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model.config.use_flash_attention = False
        self.model = self.model.to(torch.bfloat16).eval()

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
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.processor(conversations=conversation, images=pil_images, force_batchify=True).to(self.model.device)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        outputs = self.model.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=300,
            do_sample=False,
            use_cache=True
        )
        return self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
