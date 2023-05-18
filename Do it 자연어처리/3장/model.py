from transformers import BertConfig, BertModel
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base", do_lower_case=False)

sentences = ["안녕하세요", "하이!"]
features = tokenizer(
    sentences,
    max_length=10,
    padding="max_length",
    truncation=True,
)


pretrained_model_config = BertConfig.from_pretrained(
    "beomi/kcbert-base"
)

model = BertModel.from_pretrained(
    "beomi/kcbert-base",
    config=pretrained_model_config,
)

#print(pretrained_model_config)

features = { k: torch.tensor(v) for k,v in features.items() }

outputs = model(**features)

print(outputs.pooler_output)
print(outputs.pooler_output.shape)