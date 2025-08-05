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
        self.vision_adapter = nn.Linear(self.vision_encoder.config.hidden_size, self.config.hidden_size)
        self.text_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_fast=False)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def generate(self, input_ids=None, video_frames=None, attention_mask=None, **generate_kwargs):
        if video_frames is not None:
            # 埋め込みを作成
            text_embeds = self.get_input_embeddings()(input_ids)
            vision_outputs = self.vision_encoder(video_frames, output_hidden_states=True)
            video_features = self.vision_adapter(vision_outputs.pooler_output).unsqueeze(1)
            inputs_embeds = torch.cat([video_features, text_embeds], dim=1)
            if attention_mask is not None:
                video_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([video_mask, attention_mask], dim=1)
            return super().generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generate_kwargs
            )
        else:
            return super().generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs
            )