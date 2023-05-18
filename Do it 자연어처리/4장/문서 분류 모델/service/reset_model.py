import torch
from transformers import BertConfig, BertForSequenceClassification
from modelArg import *

fine_tuned_model_ckpt = torch.load(
    args.downstream_model_checkpoint_fpath,
    map_location=torch.device("cpu")
)
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
)
model = BertForSequenceClassification(pretrained_model_config)
model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
model.eval()


def inference_fn(sentence):
    inputs = tokenizer(
        [sentence],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )
    with torch.no_grad():
        outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})
        prob = outputs.logits.softmax(dim=1)
        positive_prob = round(prob[0][1].item(), 4)
        negative_prob = round(prob[0][0].item(), 4)
        pred = "긍정 (positive)" if torch.argmax(prob) == 1 else "부정 (negative)"
    return {
        'sentence': sentence,
        'prediction': pred,
        'positive_data': f"긍정 {positive_prob}",
        'negative_data': f"부정 {negative_prob}",
        'positive_width': f"{positive_prob * 100}%",
        'negative_width': f"{negative_prob * 100}%",
    }

from ratsnlp.nlpbook.classification import get_web_service_app
app = get_web_service_app(inference_fn)
app.run()