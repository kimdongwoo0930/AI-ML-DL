import os
from Korpora import Korpora
from tokenizers import ByteLevelBPETokenizer
from tokenizers import BertWordPieceTokenizer

# nsmc라는 데이터 셋을 Korpora를 이용해 불러온다.
nsmc = Korpora.load("nsmc",force_download=True)

def write_lines(path,lines):
  with open(path,'w',encoding="UTF-8") as f:
    for line in lines:
      f.write(f'{line}\n')

write_lines("./AI/Do it/토큰/content/train.txt",nsmc.train.get_all_texts())
write_lines("./AI/Do it/토큰/content/test.txt",nsmc.test.get_all_texts())

os.makedirs("./AI/Do it/토큰/nlpbook/bbpe",exist_ok=True)

bytebpe_tokenizer = ByteLevelBPETokenizer()
bytebpe_tokenizer.train(
    files=["./AI/Do it/토큰/content/train.txt", "./AI/Do it/토큰/content/test.txt"],
    vocab_size=10000,
    special_tokens=["[PAD]"]
)
bytebpe_tokenizer.save_model("./AI/Do it/토큰/nlpbook/bbpe")


os.makedirs("./AI/Do it/토큰/nlpbook/wordpiece",exist_ok=True)


wordpiece_tokenizer = BertWordPieceTokenizer(lowercase=False)
wordpiece_tokenizer.train(
    files=["./AI/Do it/토큰/content/train.txt", "./AI/Do it/토큰/content/test.txt"],
    vocab_size=10000,
)
wordpiece_tokenizer.save_model("./AI/Do it/토큰/nlpbook/wordpiece")