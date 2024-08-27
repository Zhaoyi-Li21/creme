# a testing case
# patching "The nationality of the creator of C. Auguste Dupin is __"


model_name=llama2-7b # openalpaca-3b
model_string=llama2 # openalpaca
model_path=/root/autodl-tmp/zhaoyi/huggingface_models/llama2-7b-hf # change it into yours
dir_path=/root/autodl-tmp/zhaoyi/creme/creme

CUDA_VISIBLE_DEVICES=0 python comp_editor_inspect.py \
    --data $dir_path/data/example.json \
    --model $model_path \
    --config llama-7b \
    --template default > testing.txt