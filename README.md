## CUPCase: Investigating Complex Disease Diagnosis With Large Language Models 

This is the official repository for the CUPCase paper and dataset.

This repository is comprised of three main components:
* lm_eval - for the evaluation of on premise models like llama3.1, Meditron, BioMistral
* gpt_medlm_evaluation - for the evaluation of API based LLMs like GPT-4o and Medlm-large, including a variant framework for refinement experiments
* utils, preprocess - for general utilis and preprocessing of the CUPCase dataset
To use any of the above, follow the specific readme files in each.

The dataset is available on huggingface - https://huggingface.co/datasets/ofir408/CupCase

The CUPCase paper was accepted to AAAI 2025.

Paper Link: https://ojs.aaai.org/index.php/AAAI/article/view/35050

## Citation:
If you use CupCase or find this repository useful for your research or work, please cite us using the following citation:

```
@inproceedings{perets2025cupcase,
  title={CUPCase: Clinically Uncommon Patient Cases and Diagnoses Dataset},
  author={Perets, Oriel and Shoham, Ofir Ben and Grinberg, Nir and Rappoport, Nadav},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={27},
  pages={28293--28301},
  year={2025}
}
```
