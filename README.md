# Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning

We proposed a new AMR-based logic-driven data augmentation for contrastive learning intermediate training and then we conduct the downstream tasks require logical reasoning including logical reasoning reading comprehension tasks (ReClor and LogiQA) and natural language inference tasks (MNLI, MRPC, RTE, QNLI and QQP). Our method `AMR-LE-Ensemble` model and `AMR-LE (DeBERTa-v2-xxlarge-AMR-LE-Cont)` achieves `#2` and `#5` on the [ReClor leaderboard](https://eval.ai/web/challenges/challenge-page/503/leaderboard/1347) respectively and we also release the model weights on `Huggingface/models`.

## 🤗 Model weights
[AMR-LE-DeBERTa-V2-XXLarge-Contraposition](https://huggingface.co/qbao775/AMR-LE-DeBERTa-V2-XXLarge-Contraposition)

[AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation](https://huggingface.co/qbao775/AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation)

[AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation-Implication](https://huggingface.co/qbao775/AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation-Implication)

[AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation-Implication-Commutative](https://huggingface.co/qbao775/AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation-Implication-Commutative)

[AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation-Implication-Commutative-Pos-Neg-1-2](https://huggingface.co/qbao775/AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation-Implication-Commutative-Pos-Neg-1-2)

[AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation-Implication-Commutative-Pos-Neg-1-3](https://huggingface.co/qbao775/AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation-Implication-Commutative-Pos-Neg-1-3)

## To replicate our experiment result, you can follow the following steps.
1. Install the all required packages from the requirements_latest.txt `pip install -r requirements_latest.txt`

## Logical equivalence-driven data augmentation
### Synthetic sentences generation
1. You can run `logical_equivalence_synthetic_dataset.py` to automatically generate sentences which is ready for the stage-1 finetuning.
2. All code about logical equivalence data augmentation can be found in logical_equivalence_functions.py. You can run the script by `python logical_equivalence_functions.py`
3. To adjust the porprotion of positive and negative samples in the stage-1 finetuning, you can run the `negative_sample_extention.py`.
 
 ## Logical equivalence-driven data augmentation for representation learning
 You can follow the running script `script_running_notes.txt` and use the training commands to conduct stage-1 finetuning and stage-2 finetuning. Please remember you need to conduct the stage-1 finetuning firstly and then conduct the stage-2 finetuning. The main function code is in `BERT/run_glue_no_trainer.py`.
 Here is an example of stage-1 finetuning.
 ```
 python run_glue_no_trainer.py \
  --seed 2021 \
  --model_name_or_path roberta-large \
  --train_file ../output_result/Synthetic_xfm_t5wtense_logical_equivalence_train_v4.csv \
  --validation_file ../output_result/Synthetic_xfm_t5wtense_logical_equivalence_validation_v4.csv \
  --max_length 256 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir Transformers/roberta-large-our-model-v4/
 ```
 Here is an example of stage-2 finetuning on MRPC.
 ```
 python run_glue_no_trainer.py \
  --seed 42 \
  --model_name_or_path Transformers/roberta-large-our-model-v4/ \
  --task_name mrpc \
  --max_length 256 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir Transformers/mrpc/synthetic-logical-equivalence-finetuned-roberta-large-v4/
 ```
 

 
 For the stage-2 finetuning on ReClor and LogiQA, you need to run the commands under the `BERT/scripts`. 
 Here is an example of stage-2 finetuning for ReClor.
 ```
export RECLOR_DIR=reclor_data
export TASK_NAME=reclor
export MODEL_NAME=microsoft/deberta-v2-xxlarge
export OUTPUT_NAME=deberta-v2-xxlarge

CUDA_VISIBLE_DEVICES=3 python run_multiple_choice.py \
    --model_type debertav2 \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --do_test \
    --do_lower_case \
    --data_dir $RECLOR_DIR \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 4   \
    --per_gpu_train_batch_size 4   \
    --gradient_accumulation_steps 24 \
    --learning_rate 1e-05 \
    --num_train_epochs 10.0 \
    --output_dir Checkpoints/$TASK_NAME/${OUTPUT_NAME} \
    --logging_steps 200 \
    --save_steps 200 \
    --adam_betas "(0.9, 0.98)" \
    --adam_epsilon 1e-6 \
    --no_clip_grad_norm \
    --warmup_proportion 0.1 \
    --weight_decay 0.01
  ```
 Here is an example of stage-2 finetuning for LogiQA.
  ```
export RECLOR_DIR=logiqa_data
export TASK_NAME=logiqa
export MODEL_NAME=microsoft/deberta-v2-xxlarge
export OUTPUT_NAME=deberta-v2-xxlarge

CUDA_VISIBLE_DEVICES=3 python run_multiple_choice.py \
    --model_type debertav2 \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --do_test \
    --do_lower_case \
    --data_dir $RECLOR_DIR \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 4   \
    --per_gpu_train_batch_size 4   \
    --gradient_accumulation_steps 24 \
    --learning_rate 1e-05 \
    --num_train_epochs 10.0 \
    --output_dir Checkpoints/$TASK_NAME/${OUTPUT_NAME} \
    --logging_steps 200 \
    --save_steps 200 \
    --adam_betas "(0.9, 0.98)" \
    --adam_epsilon 1e-6 \
    --no_clip_grad_norm \
    --warmup_proportion 0.1 \
    --weight_decay 0.01
  ```
