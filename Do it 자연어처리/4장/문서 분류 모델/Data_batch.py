from ratsnlp.nlpbook.classification import NsmcCorpus, ClassificationDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from modelArgs import *
from Tokenizers import *


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
    num_workers=0,
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
