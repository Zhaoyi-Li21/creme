CUDA_VISIBLE_DEVICES=0 python comp_editor_inspect.py \
    --data /root/autodl-tmp/zhaoyi/knowledge_locate/FastEdit/data/comp_example.short_cut.json \
    --model /root/autodl-tmp/zhaoyi/huggingface_models/llama2-7b-hf \
    --config llama-7b \
    --template default > result_comp_edit.short_cut.txt