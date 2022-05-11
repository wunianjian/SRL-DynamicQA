python3 ../module/ArgumentLabeling/ckpt_eval.py \
--data_path ../data/conll2012/test.english.psense.plabel.conll12.json \
--checkpoint_path ../checkpoints/conll2012/arg_labeling/2022_05_07_07_41_46/checkpoint_19.cpt \
--arg_query_type 2 \
--argm_query_type 1 \
--gold_level 1 \
--max_tokens 2048 \
--amp


