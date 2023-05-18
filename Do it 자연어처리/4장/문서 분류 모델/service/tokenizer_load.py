from transformers import BertTokenizer
from modelArg import *

tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)