from transformers import BertConfig, BertForSequenceClassification
from modelArgs import *
from Data_batch import *

pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels = corpus.num_labels,
)

model = BertForSequenceClassification.from_pretrained(
    args.pretrained_model_name,
    config=pretrained_model_config,
)