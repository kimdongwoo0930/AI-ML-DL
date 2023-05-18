from ratsnlp.nlpbook.classification import ClassificationDeployArguments
args = ClassificationDeployArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_model_dir="AI/Do it/4장/문서 분류 모델/nlpbook/checkpoint-doccls",
    max_seq_length=128,
)