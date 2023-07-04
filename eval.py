#%%
import torch
import torchsummary
from trainer import BERTTRainer
import underthesea
from preprocessing.data import DataProcessor
import io
import json
# %%
processor = DataProcessor(tokenizer_path="./tokenizer/tokenizer.pkl")
# %%
# %%
trainer = BERTTRainer(
    token_size=len(processor.dictionary),
    num_entities=len(processor.entities),
    device='cuda',
    checkpoint='./saved_models/bert.pt'
)
#%%
dataset = io.open("D:\Datasets\Vi - NER/data.json", encoding='utf-8').read().strip().split("\n")
# %%
json_data = json.loads(dataset[5])
# %%
json_data['words']
# %%
digits, _ = processor.fit(json_data['words'], train=False)
# %%
digits
# %%
digits = torch.tensor(digits).unsqueeze(0).to('cuda')
# %%
trainer.model.eval()
# %%
outputs = trainer.model(digits)
# %%
_, pred = torch.max(outputs, dim=-1)
# %%
pred
# %%
prediction = []
for entity in pred[0][1:-1]:
    prediction.append(processor.entities[entity.item()])
# %%
prediction
# %%
json_data['tags']
# %%

# %%
