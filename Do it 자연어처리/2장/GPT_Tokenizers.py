from transformers import GPT2Tokenizer

tokenizers_gpt = GPT2Tokenizer.from_pretrained("./AI/Do it/토큰/nlpbook/bbpe")
tokenizers_gpt.pad_token = "[PAD]"

sentences = [
    "아 더빙.. 진짜 짜증나네요 목소리",
    "흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나",
    "별루 였다..",
]

tokenizers_sentences = [ tokenizers_gpt.tokenize(sentence) for sentence in sentences ]

print(tokenizers_sentences)
print()


# padding : 문장의 최대 길이에 맞춰 패딩
# max_length : 문장의 토큰 기준 최대 길이
# truncation : 문장 잘림 허용 옵션
batch_inputs = tokenizers_gpt(sentences,padding="max_length",max_length=12,truncation=True)

print(batch_inputs["input_ids"])
print()

# vocab.json을 통해 토큰에 맞게 인덱스로 변환한 리스트다. ( 인덱싱 )

print(batch_inputs["attention_mask"])

# 일반토큰 패딩토큰을 구분해주는 장치이다.