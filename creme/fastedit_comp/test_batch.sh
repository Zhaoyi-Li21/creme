# test_batch: for original, paraphrased, generalization
# test_batch_v2: for irrelevant (better irrelevant)


for pat in 'suf'
do
    CUDA_VISIBLE_DEVICES=0 python comp_editor_batch.py \
        --data /root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/make_dataset/llama2.$pat.json \
        --model /root/autodl-tmp/zhaoyi/huggingface_models/llama2-7b-hf \
        --config llama-7b \
        --template default
done

for pat in 'suf'
do
    CUDA_VISIBLE_DEVICES=0 python comp_editor_batch.py \
        --data /root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/make_dataset/openalpaca.$pat.json \
        --model /root/autodl-tmp/zhaoyi/huggingface_models/openalpaca3b \
        --config llama-7b \
        --template default
done