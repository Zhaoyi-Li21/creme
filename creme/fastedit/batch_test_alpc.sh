# for pat in 'suf'
# do
#     CUDA_VISIBLE_DEVICES=0 python editor_baseline.py \
#         --data /root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/make_dataset/llama2.$pat.new.json \
#         --model /root/autodl-tmp/zhaoyi/huggingface_models/llama2-7b-hf \
#         --config llama-7b \
#         --template default
# done

for pat in 'suf'
do
    CUDA_VISIBLE_DEVICES=0 python editor_baseline.py \
        --data /root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/make_dataset/openalpaca.$pat.new.json \
        --model /root/autodl-tmp/zhaoyi/huggingface_models/openalpaca3b \
        --config llama-7b \
        --template default
done