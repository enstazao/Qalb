# Qalb-1.0

**An Urdu large language model built with Llama 3.1 8B.**

[Paper](https://arxiv.org/abs/2601.08141) · [Hugging Face model](https://huggingface.co/enstazao/Qalb-1.0-8B-Instruct) · [Training details](docs/training_details.md) · [Presentation](presentation/qalb.html) · [Apache 2.0 code license](LICENSE)

Qalb adapts an existing language model to Urdu through **continued pre-training (CPT)** followed by **supervised fine-tuning (SFT)**. This repository provides two author-supplied CPT implementations, their configurations, and practical documentation for research use. The released instruction model is available on Hugging Face.

**Paper:** *Qalb: Largest State-of-the-Art Urdu Large Language Model for 230M Speakers with Systematic Continued Pre-Training*
Muhammad Taimoor Hassan, Jawad Ahmed, and Muhammad Awais · [arXiv:2601.08141](https://arxiv.org/abs/2601.08141)

## Model and training approach

The foundation is Llama 3.1 8B, a decoder-only Transformer for causal language modeling. The released Qalb checkpoint has undergone instruction tuning; the CPT script starts from the **base** checkpoint, `unsloth/Meta-Llama-3.1-8B`.

```text
Llama 3.1 8B base → Continued pre-training → Supervised fine-tuning → Qalb-1.0-8B-Instruct
```

## Citation

```bibtex
@article{hassan2026qalb,
  title   = {Qalb: Largest State-of-the-Art Urdu Large Language Model for 230M Speakers with Systematic Continued Pre-Training},
  author  = {Hassan, Muhammad Taimoor and Ahmed, Jawad and Awais, Muhammad},
  journal = {arXiv preprint arXiv:2601.08141},
  year    = {2026},
  doi     = {10.48550/arXiv.2601.08141},
  url     = {https://arxiv.org/abs/2601.08141}
}
```
