from transformers import BertTokenizer

tokenizer_bert = BertTokenizer.from_pretrained(
    "./AI/Do it/토큰/nlpbook/wordpiece",
    do_lower_case=False,
)

sentences = [
    "아 더빙.. 진짜 짜증나네요 목소리",
    "흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나",
    "별루 였다..",
]

tokenizer_sentences = [ tokenizer_bert.tokenize(sentence) for sentence in sentences ]

print(tokenizer_sentences)
print()

# padding : 문장의 최대 길이에 맞춰 패딩
# max_length : 문장의 토큰 기준 최대 길이
# truncation : 문장 잘림 허용 옵션
batch_inputs = tokenizer_bert(sentences,padding="max_length",max_length=12,truncation=True)
print(batch_inputs["input_ids"])
print()

#[CLS], [SEP]
# BERT는 문장 시작과 끝에 2개의 토큰을 덧붙이는 특징이 있다.

print(batch_inputs["attention_mask"])
print()
# GPT와 똑같이 작동한다.

print(batch_inputs["token_type_ids"])

# 세그먼트에 해당하는 것으로 모두 0이다.
# 세그먼트 정보를 입력하는 건 BERT 모델의 특징이다. 
