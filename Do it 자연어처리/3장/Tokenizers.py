from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base", do_lower_case=False)

sentences = ["안녕하세요", "하이!"]
features = tokenizer(
    sentences,
    max_length=10,
    padding="max_length",
    truncation=True,
)

print(features)
