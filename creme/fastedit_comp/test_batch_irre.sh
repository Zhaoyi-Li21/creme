# test_batch: for original, paraphrased, generalization
# test_batch_irre: for irrelevant (better irrelevant)

model_name=llama2-7b # openalpaca-3b
model_string=llama2 # openalpaca
model_path=/root/autodl-tmp/zhaoyi/huggingface_models/llama2-7b-hf # change it into yours
dir_path=/root/autodl-tmp/zhaoyi/creme/creme

CUDA_VISIBLE_DEVICES=1 python comp_editor_batch_irre.py \
    --data $dir_path/make_dataset/$model_string.suf.irre.json \
    --model $model_path \
    --config llama-7b \
    --template default

# for pat in 'suf'
# do
#     CUDA_VISIBLE_DEVICES=1 python comp_editor_batch_irre.py \
#         --data /root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/make_dataset/llama2.$pat.irre.json \
#         --model /root/autodl-tmp/zhaoyi/huggingface_models/llama2-7b-hf \
#         --config llama-7b \
#         --template default
# done

# for pat in 'suf'
# do
#     CUDA_VISIBLE_DEVICES=1 python comp_editor_batch_irre.py \
#         --data /root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/make_dataset/openalpaca.$pat.irre.json \
#         --model /root/autodl-tmp/zhaoyi/huggingface_models/openalpaca3b \
#         --config llama-7b \
#         --template default
# done

