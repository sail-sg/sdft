#!/bin/bash
set -e
source "scripts/utils.sh"

# Configurations
model_path="/home/jovyan/Llama-2-7b-chat-hf"
cuda_visible_devices="0"
type=sft
train_dataset=dolly
output_folder="predictions/${train_dataset}/${type}"
result_file="results/${train_dataset}/${type}.log"
checkpoint_dir="checkpoints/${train_dataset}/${type}"

# Hyperparameters
epoch=2
lr=1e-4
per_device_train_batch_size=2

create_empty_file ${result_file}
echo -e "Fine-tuning using ${type}\n" >> ${result_file}

CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python main.py \
    --stage sft \
    --model_name_or_path ${model_path} \
    --do_train \
    --dataset ${train_dataset}_train \
    --template alpaca \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ${checkpoint_dir} \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate ${lr} \
    --num_train_epochs ${epoch} \
    --plot_loss \
    --bf16

# Evaluate math reasoning capabilities
for math_dataset in gsm8k multiarith;
do
    echo "Evaluation on ${math_dataset}:" >> ${result_file}
    output_dir="${output_folder}/${math_dataset}"
    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python main.py \
        --stage sft \
        --model_name_or_path ${model_path} \
        --adapter_name_or_path ${checkpoint_dir} \
        --do_predict \
        --dataset "${math_dataset}_test" \
        --template gsm8k_infer \
        --output_dir ${output_dir} \
        --per_device_eval_batch_size 1 \
        --max_samples 9999999999999 \
        --predict_with_generate \
        --overwrite_cache \
        --fp16

    python "eval/eval_math.py" --input_file "${output_dir}/generated_predictions.jsonl" >> ${result_file}
done

# Evaluate on OpenFunctions
echo "Evaluation on OpenFunctions:" >> ${result_file}
output_dir="${output_folder}/openfunction"
CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python main.py \
    --stage sft \
    --model_name_or_path ${model_path} \
    --adapter_name_or_path ${checkpoint_dir} \
    --do_predict \
    --dataset openfunction_test \
    --template alpaca \
    --output_dir ${output_dir} \
    --per_device_eval_batch_size 1 \
    --max_samples 9999999999999 \
    --predict_with_generate \
    --overwrite_cache \
    --fp16

python "eval/eval_openfunction.py" --input_file "${output_dir}/generated_predictions.jsonl" >> ${result_file}

# Evaluate on HumanEval
output_path="${output_folder}/humaneval/result.json"
create_empty_file ${output_path}
CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python bigcode-evaluation-harness/main.py \
    --model ${model_path} \
    --peft_model ${checkpoint_dir} \
    --tasks humanevalsynthesize-python \
    --prompt octocoder \
    --do_sample False \
    --batch_size 1 \
    --allow_code_execution \
    --trust_remote_code \
    --metric_output_path ${output_path} \
    --max_length_generation 2048 \
    --precision fp16

python "eval/eval_humaneval.py" --input_file ${output_path} >> ${result_file}

# Predict on alpaca_eval (general helpfulness)
output_dir="${output_folder}/alpaca_eval"
CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python main.py \
    --stage sft \
    --model_name_or_path ${model_path} \
    --adapter_name_or_path ${checkpoint_dir} \
    --do_predict \
    --dataset alpaca_eval \
    --template alpaca \
    --output_dir ${output_dir} \
    --per_device_eval_batch_size 1 \
    --max_samples 9999999999999 \
    --predict_with_generate \
    --overwrite_cache \
    --fp16

python "eval/prepare_alpaca_eval.py" --input_file "${output_dir}/generated_predictions.jsonl" --output_file "${output_dir}/outputs.json"
# Execute the line below yourself if you want. Configuration of OpenAI API needed. The evaluation takes about $15.
# alpaca_eval --model_outputs "${output_dir}/outputs.json"

# Evaluate safety
for template in "alpaca" "alpaca_gcg";
do
    if [ ${template} == "alpaca" ];
    then
        safety_type="raw"
    else
        safety_type="jailbreak"
    fi
    echo "Evaluation on ${safety_type} safety:" >> ${result_file}
    output_dir="${output_folder}/advbench-${safety_type}"
    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python main.py \
        --stage sft \
        --model_name_or_path ${model_path} \
        --adapter_name_or_path ${checkpoint_dir} \
        --do_predict \
        --dataset advbench \
        --template ${template} \
        --output_dir ${output_dir} \
        --per_device_eval_batch_size 1 \
        --max_samples 9999999999999 \
        --predict_with_generate \
        --overwrite_cache \
        --fp16
    
    python "eval/keyword_eval_safety.py" --input_file "${output_dir}/generated_predictions.jsonl" >> ${result_file}
done

# Evaluate general knowledge 
output_dir="${output_folder}/lm-eval"
lm_eval --model hf \
    --model_args "pretrained=${model_path}" \
    --tasks mmlu,truthfulqa,ai2_arc,hellaswag,winogrande \
    --device "cuda:${cuda_visible_devices}" \
    --batch_size 1 \
    --output_path ${output_dir}

python "eval/eval_general_knowledge.py" --input_file "${output_dir}/results.json" >> ${result_file}
echo "Evaluation of the seed LM finished successfully. Results are saved in ${result_file}."