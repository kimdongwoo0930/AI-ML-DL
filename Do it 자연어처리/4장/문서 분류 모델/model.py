from modelArgs import *
from Reset_Model import *
from Data_batch import *
from ratsnlp.nlpbook.classification import ClassificationTask

task = ClassificationTask(model, args)

trainer = nlpbook.get_trainer(args)

trainer.fit(
    task,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)