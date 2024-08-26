# v3 is for collecting generalization errors

for pat in 'suf'
do
    CUDA_VISIBLE_DEVICES=0 python comp_editor_batch_v3.py \
        --data /root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/make_dataset/llama2.$pat.json \
        --model /root/autodl-tmp/zhaoyi/huggingface_models/llama2-7b-hf \
        --config llama-7b \
        --template default
done