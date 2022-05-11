python ./module/ArgumentLabelingDQA/train.py \
--dataset_tag conll2012 \
--pretrained_model_name_or_path deepset/roberta-large-squad2 \
--train_path ./data/conll2012/train.english.psense.plabel.conll12.json \
--dev_path ./data/conll2012/dev.english.psense.plabel.conll12.json  \
--max_tokens 514 \
--max_epochs 3 \
--lr 2e-5 \
--max_grad_norm 1 \
--warmup_ratio 0.01 \
--gold_level 1 \
--tensorboard \
--amp \
--tqdm_mininterval 500 \
>log_al_qa.txt
#cat log_al.txt
