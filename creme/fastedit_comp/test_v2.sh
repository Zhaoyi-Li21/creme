CUDA_VISIBLE_DEVICES=0 python comp_editor_v2.py \
    --data /root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/data/comp_example.3.json \
    --model /root/autodl-tmp/zhaoyi/huggingface_models/llama2-7b-hf \
    --config llama-7b \
    --template default > result_comp_edit.incomplete_reasoning.txt