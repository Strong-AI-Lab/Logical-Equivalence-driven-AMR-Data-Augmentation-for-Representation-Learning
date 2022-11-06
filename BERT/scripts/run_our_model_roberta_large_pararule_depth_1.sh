export RECLOR_DIR=PARARULE_Plus
export TASK_NAME=pararule_depth_1
export MODEL_NAME=../amrlib-master/BERT/Transformers/roberta-large-our-model-v4
export OUTPUT_DIR=our-model-roberta-large-pararule-v4

CUDA_VISIBLE_DEVICES=5 python run_multiple_choice.py \
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
    --output_dir Checkpoints/pararule-plus/$OUTPUT_DIR \
    --fp16 \
    --logging_steps 200 \
    --save_steps 200 \
    --adam_betas "(0.9, 0.98)" \
    --adam_epsilon 1e-6 \
    --no_clip_grad_norm \
    --warmup_proportion 0.1 \
    --weight_decay 0.01



export RECLOR_DIR=PARARULE_Plus
export TASK_NAME=pararule_depth_1_reparaphrased
export MODEL_NAME=../amrlib-master/BERT/Transformers/roberta-large-our-model-v4
export OUTPUT_DIR=our-model-roberta-large-pararule-logical-equivalence-v4

CUDA_VISIBLE_DEVICES=5 python run_multiple_choice.py \
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
    --output_dir Checkpoints/pararule-plus/$OUTPUT_DIR \
    --fp16 \
    --logging_steps 200 \
    --save_steps 200 \
    --adam_betas "(0.9, 0.98)" \
    --adam_epsilon 1e-6 \
    --no_clip_grad_norm \
    --warmup_proportion 0.1 \
    --weight_decay 0.01