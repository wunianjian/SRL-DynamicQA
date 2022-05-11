python ./module/ArgumentLabelingDQA/ckpt_eval.py \
--checkpoint_path ./dqa_results/checkpoint-85500 \
--data_path ./data/conll2012/dev.english.psense.plabel.conll12.json \
--gold_level 1 \
--max_tokens 514 \
--amp
