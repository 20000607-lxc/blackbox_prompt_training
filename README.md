# blackbox tuning for pretrained language model
`run_glue_blackbox.py` contains the code to train  black-box prompt， `run_glue.py` is the baseline(whitebox) implementation.

## datasets 
GLUE benchmarks, such as cola, mrpc

##language model
roberta-base

## transformers
please use transformers 4.6.0

## black-box training method:
from repo: https://github.com/TransEmbedBA/TREMBA 


## how to run:
specify task_name **or** the file_name

e.g. 

`python run_glue_blackbox.py --task_name=mrpc`

`python run_glue.py --task_name=mrpc`
