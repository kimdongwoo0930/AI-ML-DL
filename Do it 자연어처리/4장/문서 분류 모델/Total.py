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
    #tpu_cores=0 if torch.cuda.is_available() else 8,
    tpu_cores=0,
    seed=7,
    
)

nlpbook.set_seed(args)
nlpbook.set_logger(args)

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)





from ratsnlp.nlpbook.classification import NsmcCorpus, ClassificationDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler



corpus = NsmcCorpus()
train_dataset = ClassificationDataset(
    args = args, 
    tokenizer = tokenizer,
    corpus = corpus, 
    mode="train"
)
print(train_dataset[100])

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=RandomSampler(train_dataset, replacement=False),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)

print(train_dataloader)

val_dataset = ClassificationDataset(
    args = args, 
    tokenizer = tokenizer, 
    corpus= corpus,
    mode="test"
)
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=args.batch_size, 
    sampler=SequentialSampler(val_dataset), 
    collate_fn=nlpbook.data_collator, 
    drop_last=False, 
    num_workers=args.cpu_workers,
)

print(val_dataset)
print(val_dataloader)


from transformers import BertConfig, BertForSequenceClassification

pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels = corpus.num_labels,
)

model = BertForSequenceClassification.from_pretrained(
    args.pretrained_model_name,
    config=pretrained_model_config,
)


from ratsnlp.nlpbook.classification import ClassificationTask

task = ClassificationTask(model, args)

trainer = nlpbook.get_trainer(args)

trainer.fit(
    task,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)