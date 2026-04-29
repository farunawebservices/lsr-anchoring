# Latent Space Refusal Anchoring for Low-Resource African Languages: Mechanistic Safety Recovery Without Retraining

A training-free method for recovering safety alignment in multilingual LLMs using Mean Activation Steering and
Sparse Autoencoder (SAE)-derived refusal direction anchors applied at inference time.

## Overview

Large Language Models fine-tuned for safety in English frequently exhibit **alignment
gaps** when prompted in low-resource languages. This project investigates whether
English-derived refusal directions extracted via mean activation steering and SAEs from the residual stream
can be steered back into the model at inference time to recover safe behaviour across
6 languages and 4 model families, with no fine-tuning required.

## Languages

| Language | Family | Resource Level |
|----------|--------|----------------|
| Yoruba | Niger-Congo | Low |
| Hausa | Afro-Asiatic | Low |
| Igbo | Niger-Congo | Low |
| Igala | Niger-Congo | Very Low |
| Swahili | Bantu | Mid |
| Arabic | Semitic | High |

## Models

| Model | Anchor Layer | Method |
|-------|-------------|--------|
| `meta-llama/Llama-3.1-8B-Instruct` | 12 | Path A (SAE) + Path B (Mean-Act) |
| `meta-llama/Llama-3.1-70B-Instruct` | 26 | Path B (Mean-Act) |
| `mistralai/Mistral-7B-Instruct-v0.3` | 16 | Path B (Mean-Act) |
| `Qwen/Qwen2.5-7B-Instruct` | 26 | Path B (Mean-Act) |

## Repository Structure

lsr-anchoring/
├── experiments/
│ ├── experiment_8b_pathA.py # Llama-3.1 8B — SAE steering
│ ├── experiment_8b_pathB.py # Llama-3.1 8B — Mean-activation steering
│ ├── experiment_70b.py # Llama-3.1 70B
│ ├── experiment_mistral7b.py # Mistral-7B-Instruct-v0.3
│ ├── experiment_qwen7b.py # Qwen2.5-7B-Instruct
│ └── patch_experiment.py # Utility patch script
├── prompts/
│ ├── prompts_v2.py # Full prompt set (100 harmful + 50 benign × 6 langs)
│ └── prompts_v2_mistral.py # Mistral-adapted prompt set
├── results/
│ └── README.md # Pointer to HuggingFace dataset
├── requirements.txt
└── README.md


## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your HuggingFace token

```bash
export HF_TOKEN=your_token_here
huggingface-cli login
```

### 3. Run an experiment

```bash
# Llama-3.1 8B — SAE steering (Path A)
python experiments/experiment_8b_pathA.py

# Llama-3.1 70B
python experiments/experiment_70b.py

# Mistral-7B
python experiments/experiment_mistral7b.py

# Qwen2.5-7B
python experiments/experiment_qwen7b.py
```

## Evaluation Setup

- **Harmful prompts:** 100 per language
- **Benign prompts:** 50 per language
- **Alpha sweep:** α ∈ {5, 10, 20, 30, 40, 50, 60, 70}
- **Metrics:** SRR (Steered Refusal Rate), DPL (Delta Precision Loss), KL divergence

## Key Results

| Model | Best Language | Peak SRR | KL | DPL |
|-------|--------------|----------|----|-----|
| Llama-3.1 8B (SAE) | Igala | 0.844 | 0.31 | 0.06 |
| Llama-3.1 8B (Mean-Act) | Hausa | 0.71 | 0.48 | 0.12 |
| Llama-3.1 70B | Hausa | 0.62 | 0.29 | 0.08 |
| Mistral-7B | Igbo | 0.58 | 0.22 | 0.04 |
| Qwen2.5-7B | Hausa | 0.51 | 0.49 | 0.50 |

> **Notable:** Arabic shows inverse transfer on Qwen2.5-7B, the English refusal
> direction actively reduces refusal rates as alpha increases (SRR → −0.10 at α=70),
> indicating English SAE directions do not generalise to Arabic in this model family.

## Full Results & Logs

All results, anchor caches, and experiment logs are hosted on HuggingFace:
**https://huggingface.co/datasets/Faruna01/lsr-anchoring-phase2-results**

## Citation

```bibtex
@misc{faruna2026lsranchoring,
  title     = {Latent Space Refusal Anchoring for Low-Resource African Languages:
Mechanistic Safety Recovery Without Retraining},
  author    = {Godwin, Abuh Faruna},
  year      = {2026},
  url       = {https://github.com/farunawebservices/lsr-anchoring}
}
```

## License

MIT License
