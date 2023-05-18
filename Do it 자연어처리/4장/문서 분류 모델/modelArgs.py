import torch
from ratsnlp.nlpbook.classification import ClassificationTrainArguments
from ratsnlp import nlpbook
from Korpora import Korpora

args = ClassificationTrainArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_corpus_name="nsmc",
    downstream_model_dir="./AI/Do it/4장/문서 분류 모델/nlpbook/checkpoint-doccls",
    downstream_corpus_root_dir="./AI/Do it/4장/문서 분류 모델/content",
    batch_size=32 if torch.cuda.is_available() else 4,
    learning_rate=5e-5,
    max_seq_length=128,
    epochs=3,
    #tpu_cores=0 if torch.cuda.is_available() else 6,
    tpu_cores=0,
    seed=7,
)

nlpbook.set_seed(args)
nlpbook.set_logger(args)

