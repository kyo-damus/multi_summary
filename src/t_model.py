import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPVisionModel, LlamaForCausalLM, LlamaConfig

class SummaryModel(LlamaForCausalLM):
    def __init__(self, config):
        llm_model_name = config['summary_model']['base_llm_model']
        vision_model_name = config['summary_model']['vision_encoder']
        llm_config = LlamaConfig.from_pretrained(llm_model_name)
        super().__init__(llm_config)

        # Llamaの重みをロード
        self.load_state_dict(LlamaForCausalLM.from_pretrained(llm_model_name).state_dict())

        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.vision_adapter = nn.Linear(
            self.vision_encoder.config.hidden_size,
            self.config.hidden_size
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_fast=False)

    def forward(self, video_frames, input_ids, attention_mask=None, **kwargs):
        # テキスト埋め込み
        text_embeds = self.get_input_embeddings()(input_ids)  # (B, T, H)
        # 映像特徴抽出・変換
        vision_outputs = self.vision_encoder(video_frames, output_hidden_states=True)
        video_features = self.vision_adapter(vision_outputs.pooler_output).unsqueeze(1)  # (B, 1, H)
        # 結合
        inputs_embeds = torch.cat([video_features, text_embeds], dim=1)  # (B, T+1, H)
        # attention_mask調整
        if attention_mask is not None:
            video_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([video_mask, attention_mask], dim=1)  # (B, T+1)
        # Llama本体へ
        outputs = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
        return outputs