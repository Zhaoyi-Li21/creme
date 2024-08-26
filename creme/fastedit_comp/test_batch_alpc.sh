for pat in 'pre'
do
    CUDA_VISIBLE_DEVICES=0 python comp_editor_batch.py \
        --data /root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/make_dataset/openalpaca.$pat.json \
        --model /root/autodl-tmp/zhaoyi/huggingface_models/openalpaca3b \
        --config llama-7b \
        --template default
done