import yaml
import torch
from src.t_model import SummaryModel

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
model = SummaryModel(config)

dummy_text = ["これはテストです"]
dummy_video_frames = torch.randn(1, 3, 336, 336) # (バッチ, 色, 高さ, 幅)

# 3. テキストをtokenize
tokenized = model.text_tokenizer(dummy_text, return_tensors='pt', padding=True)
input_ids = tokenized.input_ids
attention_mask = tokenized.attention_mask

# 4. カスタムの.generate()メソッドを呼び出す
outputs = model.generate(
    input_ids=input_ids,
    video_frames=dummy_video_frames,
    attention_mask=attention_mask,
    max_length=128,
    num_beams=4,
    early_stopping=True,
    do_sample=True,
    top_p=0.6    
)

# 5. 結果をデコードして表示
summaries = model.text_tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(summaries)