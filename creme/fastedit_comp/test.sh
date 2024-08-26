# a testing case
# patching "The nationality of the creator of C. Auguste Dupin is __"

CUDA_VISIBLE_DEVICES=0 python comp_editor_inspect.py \
    --data /root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/data/example.json \
    --model /root/autodl-tmp/zhaoyi/huggingface_models/llama2-7b-hf \
    --config llama-7b \
    --template default > result_comp_edit.txt