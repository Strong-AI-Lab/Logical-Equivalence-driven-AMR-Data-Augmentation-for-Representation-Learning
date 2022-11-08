export RECLOR_DIR=logiqa_data
export TASK_NAME=logiqa
export MODEL_NAME=Transformers/roberta-large-contraposition/
export OUTPUT_NAME=roberta-large-contraposition

CUDA_VISIBLE_DEVICES=0 python run_multiple_choice.py \
    --model_type roberta \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --do_test \
    --do_lower_case \
    --data_dir $RECLOR_DIR \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 8   \
    --per_gpu_train_batch_size 8   \
    --gradient_accumulation_steps 1 \
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

########################################################
export RECLOR_DIR=logiqa_data
export TASK_NAME=logiqa
export MODEL_NAME=Transformers/roberta-large-contraposition-double-negation/
export OUTPUT_NAME=roberta-large-contra-double-neg

CUDA_VISIBLE_DEVICES=2 python run_multiple_choice.py \
    --model_type roberta \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --do_test \
    --do_lower_case \
    --data_dir $RECLOR_DIR \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 8   \
    --per_gpu_train_batch_size 8   \
    --gradient_accumulation_steps 1 \
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

########################################################
export RECLOR_DIR=logiqa_data
export TASK_NAME=logiqa
export MODEL_NAME=Transformers/roberta-large-contraposition-double-negation-implication-filled/
export OUTPUT_NAME=roberta-large-contra-double-neg-impli

CUDA_VISIBLE_DEVICES=4 python run_multiple_choice.py \
    --model_type roberta \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --do_test \
    --do_lower_case \
    --data_dir $RECLOR_DIR \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 8   \
    --per_gpu_train_batch_size 8   \
    --gradient_accumulation_steps 1 \
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

########################################################
export RECLOR_DIR=logiqa_data
export TASK_NAME=logiqa
export MODEL_NAME=Transformers/roberta-large-our-model-v4-pos-neg-1-2/
export OUTPUT_NAME=roberta-large-le-our-model-logiqa-v4-pn-1-2

CUDA_VISIBLE_DEVICES=1 python run_multiple_choice.py \
    --model_type roberta \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --do_test \
    --do_lower_case \
    --data_dir $RECLOR_DIR \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 8   \
    --per_gpu_train_batch_size 8   \
    --gradient_accumulation_steps 1 \
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

########################################################
export RECLOR_DIR=logiqa_data
export TASK_NAME=logiqa
export MODEL_NAME=Transformers/roberta-large-our-model-v4-pos-neg-1-3/
export OUTPUT_NAME=roberta-large-le-our-model-logiqa-v4-pn-1-3

CUDA_VISIBLE_DEVICES=4 python run_multiple_choice.py \
    --model_type roberta \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --do_test \
    --do_lower_case \
    --data_dir $RECLOR_DIR \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 8   \
    --per_gpu_train_batch_size 8   \
    --gradient_accumulation_steps 1 \
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