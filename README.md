# CREME
This repo contains the official implementation for the paper *Understanding and Patching Compositional Reasoning in LLMs* (ACL'2024, Findings)

## Install conda environment and packages

## Data and Models
For data please switch the working path into the `./data` dictionary and run `cd mquake` command.
- The original MQuAKE-CF data (2hop split): `MQuAKE-CF-3k.2hop.json`. This data file is directly used for inference experiments (compositionality gap).
- For the inference experiments, to align the LLMs' output format to the answer space (i.e., directly output the answer rather than other common words (e.g., "OK", "Cool", repeating the question and stuff)), we use few-shot prompts to instruct models. The templates could be found in `./data/mquake/prompts`.
- For inspecting, causality and locating experiments: we use `comp_cloze_prefix.json` (or `comp_cloze_suffix.json`), where we paraphrase knowledge items in the Cloze-Test form (i.e., (subject, relation, object): The creator of C. Auguste Dupin is __ (waiting for completion), in align with previous works, e.g., [ROME](https://arxiv.org/abs/2202.05262), [Memory Injections](https://arxiv.org/abs/2309.05605), [Dissecting Factual Recall](https://arxiv.org/abs/2304.14767) and stuff).
- For editing (patching) experiments: we construct `MQuAKE-CF-3k.2hop.edit.json`, where we sample paraphrasing set, generalization set and irrelavance set (please refer to the paper for detailed introduction) for each testing case on the basis of `MQuAKE-CF-3k.2hop.json`.

## Inference Experiments (Compositionality Gap)
- For running inference experiments, please first running the command `cd inference` to switch working dictionary to the `inference` dictionary.
## Logit Lens Experiments

## Intervention Experiments

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