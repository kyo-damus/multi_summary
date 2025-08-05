import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel

class SummaryModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        llm_model_name=config['summary_model']['base_llm_model']
        vision_model_name = config['summary_model']['vision_encoder']

        self.text_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.decoder_llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)

        self.vision_adapter = nn.Linear(
            self.vision_encoder.config.hidden_size,
            self.decoder_llm.config.hidden_size
        )

    def get_input_embeddings(self, input_ids):
        return self.decoder_llm.get_input_embeddings()(input_ids)


    def forward(self, video_frames , input_ids, attention_mask=None):
        text_embeds = self.decoder_llm.get_input_embeddings()(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
        vision_outputs = self.vision_encoder(video_frames, output_hidden_states=True)
        vision_features = self.vision_adapter(vision_outputs.pooler_output).unsqueeze(1)

        embeds = torch.cat([vision_features, text_embeds], dim=1)

        if attention_mask is not None:
            batch_size = attention_mask.size(0)
            video_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([video_mask, attention_mask], dim=1)
        else:
            attention_mask = None

        outputs = self.decoder_llm(
            inputs_embeds=embeds, attention_mask=attention_mask
        )

        return outputs

    def prepare_inputs_for_generation(
            self, input_ids,
            video_spatio_temporal_features,
            past_key_values=None,
            **kwargs
        ):
        if past_key_values is None:
            # 最初のステップでは、映像特徴量が渡されるので、映像特徴量とテキストのEmbeddingを準備
            vision_features = video_spatio_temporal_features
            text_embeds = self.get_input_embeddings(input_ids)

            inputs_embeds = torch.cat([vision_features, text_embeds], dim=1)
        else:
            # 2ステップ目以降は、直前に生成された単語のIDだけをEmbeddingに変換すれば良い
            inputs_embeds = self.get_input_embeddings(input_ids)

        return {
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        }

    def generate(
            self,
            input_ids,
            video_frames,
            attention_mask=None,
            **generate_kwargs
        ):
        # 映像から特徴量を抽出・変換する
        with torch.no_grad():
            vision_outputs = self.vision_encoder(video_frames, output_hidden_states=True)
            video_features = self.vision_adapter(vision_outputs.pooler_output).unsqueeze(1)

        return self.decoder_llm.generate(
            input_ids=input_ids,
            video_spatio_temporal_features=video_features,
            attention_mask=attention_mask,
            prepare_inputs_for_generation=self.prepare_inputs_for_generation,
            **generate_kwargs
        )