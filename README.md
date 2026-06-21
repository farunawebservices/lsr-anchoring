# LSR Anchoring - Multilingual Jailbreak Benchmark & Safety Steering

A two-part project on multilingual AI safety for low-resource African languages:

1. **Benchmark** - measuring the cross-lingual safety gap across 7 languages
   and 4 open-weight models (GlobalSouthML Hackathon 2026, Apart Research)
2. **Mitigation** - LSR-Anchoring, a training-free activation-steering method
   that recovers safety at inference time with no retraining and no
   target-language data 

**Live dashboard:** https://huggingface.co/spaces/Faruna01/lsr-dashboard
**Results dataset:** https://huggingface.co/datasets/Faruna01/lsr-anchoring-phase2-results

---

## Languages

| Language | Family | Resource Level |
|----------|--------|----------------|
| Yoruba | Niger-Congo | Low |
| Hausa | Afro-Asiatic | Low |
| Igbo | Niger-Congo | Low |
| Igala | Niger-Congo | Very Low |
| Swahili | Bantu | Mid |
| Arabic | Semitic | High |

---

## Models

| Model | Used In | Anchor Layer | Method |
|-------|---------|-------------|--------|
| `meta-llama/Meta-Llama-3-8B-Instruct` | Both | 12 | Benchmark + SAE & Mean-Act |
| `meta-llama/Llama-3.1-70B-Instruct` | Steering | 26 | Mean-Act |
| `mistralai/Mistral-7B-Instruct-v0.3` | Both | 16 | Benchmark + Mean-Act |
| `Qwen/Qwen2.5-7B-Instruct` | Both | 26 | Benchmark + Mean-Act |
| `google/gemma-2-9b-it` | Benchmark | — | Benchmark only |

---

## Repository Structure
lsr-anchoring/
├── Benchmark Code/ ← Benchmark pipeline
│ └── prompts_v2-1.py — Full prompt set (100 harmful + 50 benign × 7 langs)
│
├── experiments/ ← LSR-Anchoring steering experiments
│ ├── experiment_8b_pathA.py — Llama-3.1 8B, SAE steering
│ ├── experiment_8b_pathB.py — Llama-3.1 8B, Mean-activation steering
│ ├── experiment_70b.py — Llama-3.1 70B
│ ├── experiment_mistral7b.py — Mistral-7B-Instruct-v0.3
│ ├── experiment_qwen7b.py — Qwen2.5-7B-Instruct
│ └── patch_experiment.py — Utility patch script
│
├── prompts/ ← Original LSR-Anchoring prompt sets
│ ├── prompts_v2.py
│ └── prompts_v2_mistral.py
│
├── results/
│ └── README.md — Pointer to HuggingFace dataset
│
├── requirements.txt
└── README.md

---

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

### 3. Run the benchmark

```bash
python "Benchmark Code/prompts_v2-1.py"
```

### 4. Run LSR-Anchoring steering experiments

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

---

## Study 1 — Benchmark Results

| Model | English | Yoruba | Hausa | Igbo | Igala | Swahili | Arabic |
|-------|---------|--------|-------|------|-------|---------|--------|
| Llama-3-8B | 99% | 24% | 24% | 31% | 21% | 38% | 87% |
| Mistral-7B | 85% | 11% | 28% | 15% | 16% | 25% | 12% |
| Qwen2.5-7B | 95% | 4% | 6% | 14% | 13% | 0% | 83% |
| Gemma2-9B | 97% | 55% | 72% | 20% | 47% | 94% | 99% |

---

## Study 2 — LSR-Anchoring Results

| Model | Best Language | Peak SRR | KL | DPL |
|-------|--------------|----------|----|-----|
| Llama-3.1 8B (SAE) | Igala | 0.844 | 0.31 | 0.06 |
| Llama-3.1 8B (Mean-Act) | Hausa | 0.71 | 0.48 | 0.12 |
| Llama-3.1 70B | Yoruba, Igala | 1.00 | 2.58–3.53 | — |
| Mistral-7B | Igala | 0.75 | 0.61 | 0.06 |
| Qwen2.5-7B | Hausa | 0.51 | 0.49 | 0.50 |

> **Arabic warning:** English-derived steering directions reduce refusal rates
> on Arabic harmful prompts on every architecture tested. Do not apply
> LSR-Anchoring to Arabic-language agents without language-specific
> direction derivation.

---

## Full Results & Logs

All results, benchmark CSVs, anchor caches, and experiment logs:
**https://huggingface.co/datasets/Faruna01/lsr-anchoring-phase2-results**

---

## Citation

```bibtex
@misc{faruna2026benchmark,
  title   = {Multilingual Jailbreak Vulnerability Benchmark and Mitigation
             for Low-Resource African Languages},
  author  = {Faruna, Godwin Abuh},
  year    = {2026},
  url     = {https://github.com/farunawebservices/lsr-anchoring}
}

@misc{faruna2026lsranchoring,
  title   = {Latent Space Refusal Anchoring for Low-Resource African Languages:
             Mechanistic Safety Recovery Without Retraining},
  author  = {Faruna, Godwin Abuh},
  year    = {2026},
  url     = {https://github.com/farunawebservices/lsr-anchoring}
}
```

---

## License

MIT License
