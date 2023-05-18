from Korpora import Korpora
from modelArgs import *

Korpora.fetch(
    corpus_name = args.downstream_corpus_name,
    root_dir = args.downstream_corpus_root_dir,
    force_download=True,    
)