# Understanding and Patching Compositional Multihop Reasoning in Large Language Models
This repo contains the official implementation for the paper *Understanding and Patching Compositional Reasoning in LLMs* (ACL'2024, Findings)

## Install conda environment and packages

## Data and Models
For data please switch the working path into the `./data` dictionary and run `cd mquake` command.
- The original MQuAKE-CF data (2hop split): `MQuAKE-CF-3k.2hop.json`. This data file is directly used for inference experiments (compositionality gap).
- For the inference experiments, to align the LLMs' output format to the answer space (i.e., directly output the answer rather than other common words (e.g., "OK", "Cool", repeating the question and stuff)), we use few-shot prompts to instruct models. The templates could be found in `./data/mquake/prompts`.
- For inspecting, causality and locating experiments: we use `comp_cloze_prefix.json` (or `comp_cloze_suffix.json`), where we paraphrase knowledge items in the Cloze-Test form (i.e., (subject, relation, object): The creator of C. Auguste Dupin is __ (waiting for completion), in align with previous works, e.g., [ROME](https://arxiv.org/abs/2202.05262), [Memory Injections](https://arxiv.org/abs/2309.05605), [Dissecting Factual Recall](https://arxiv.org/abs/2304.14767) and stuff).
- For editing (patching) experiments: we construct `MQuAKE-CF-3k.2hop.edit.json`, where we sample paraphrasing set, generalization set and irrelavance set (please refer to the paper for detailed introduction) for each testing case on the basis of `MQuAKE-CF-3k.2hop.json`.

## Inference Experiments (Compositionality Gap)
To run inference experiments (Compositionality Gap, Compositional Reasoning Errors), please change the working path into the `inference` dictionary (`cd inference/MQuAKE`).
- To run inference for single-hop questions, run `python inference_single.py <model_name>`, where `<model_name>` could be `llama2-7b`, `llama2-13b` or `openalpace-3b`. After finishing running the inference program, there will be a result file (`<model_name>.json`) automatically stored in the `inference/MQuAKE/single-hop` dictionary.
- To run inference for compositional two-hop questions, run `python inference_comp.py <model_name>`. After finishing running the inference program, there will be a result file (`<model_name>.json`) automatically stored in the `inference/MQuAKE/compositional` dictionary.
- After fetching the inference results for both single-hop questions and compositional two-hop questions, we can run `python filter.py <model_name> <fix_type>` to classify results into two categories: (1) `pass_all` which means that the LLM can correctly answer both single-hop questions and the corresponding compositional ones; (2) `pass_singles_fail_comp` which means that the LLM though correctly both single-hop questions, fail to solve the compositional ones (Regarding the [Compositionality Gap](https://aclanthology.org/2023.findings-emnlp.378/), Compositional Reasoning Errors). Both these two parts of results will be seperately stored into two files in the `inference/MQuAKE/filter` dictionary.
- Notes: `<fix_type>` could be `prefix` or `suffix`, indicating two different orders when composing two single-hop questions. This is for furture usage. Besides, in each single testing case, there are three paraphrasing compositional questions (share the same meaning) to test the model. Following the original [MQuAKE](https://arxiv.org/abs/2305.14795) paper, we regard the model pass the testing as long as it can correctly answer one of the three paraphrased questions.

## Logit Lens Experiments
The following three parts (*logit lens inspection, intervention experiments and locating experiments*) are in the `inspecting_and_intervention` dictionary, which was implemented highly on the basis of [ROME's official implementation](https://github.com/kmeng01/rome) (this is an acknowledgement!).
- To run Logit Lens examples, please switch the working path: `cd inspecting_and_intervention` and run the program: `python logit_lens.py`. Note that the testing example is hard-coded in the program (so we need to mannually modify the program to test different cases). Successful running the program will generate a logit lens curve figure in the `inspecting_and_intervention/logit_lens/results` dictionary.
## Intervention Experiments
To run causal intervention experiments, first change the working path `cd inspecting_and_intervention/causal_intervention`.
- First fetch causal intervention data: run the command `python fetch.py <fix_type> <model_name>`, where `<fix_type>` could be `prefix` or `suffix`; `<model_name>` could be `llama2-7b` or `openalpaca-3b`. This program will fetch intervention data and organize them as a file `<model_name>.<fix_type>.json` in the current dictionary.
- To run the causal intervention experiment: run the command `python causality.py <model_name> <fix_type>`. This program will generate a result file `<model_name>.<fix_type>.json` in the `results` dictionary.
- To aggregate the results (average over instances) and visualize them: first switch the working path into the `results` dictionary `cd results` and run the command `python aggregate_visualize.py <model_name> <fix_type>`. Successful running will generate a heatmap figure in the same dictionary.
## Locating Experiments

## Correcting Compositional Reasoning Errors via Model Editing

## Citation
If you find the paper or the repo is helpful, it would be lovely that you considering cite the paper:
```
@inproceedings{li-etal-2024-understanding,
    title = "Understanding and Patching Compositional Reasoning in {LLM}s",
    author = "Li, Zhaoyi  and
      Jiang, Gangwei  and
      Xie, Hong  and
      Song, Linqi  and
      Lian, Defu  and
      Wei, Ying",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.576",
    pages = "9668--9688",
    abstract = "LLMs have marked a revolutonary shift, yet they falter when faced with compositional reasoning tasks. Our research embarks on a quest to uncover the root causes of compositional reasoning failures of LLMs, uncovering that most of them stem from the improperly generated or leveraged implicit reasoning results. Inspired by our empirical findings, we resort to Logit Lens and an intervention experiment to dissect the inner hidden states of LLMs. This deep dive reveals that implicit reasoning results indeed surface within middle layers and play a causative role in shaping the final explicit reasoning results. Our exploration further locates multi-head self-attention (MHSA) modules within these layers, which emerge as the linchpins in accurate generation and leveraing of implicit reasoning results. Grounded on the above findings, we develop CREME, a lightweight method to patch errors in compositional reasoning via editing the located MHSA modules. Our empirical evidence stands testament to CREME{'}s effectiveness, paving the way for autonomously and continuously enhancing compositional reasoning capabilities in language models.",
}
```