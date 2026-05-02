# Leduc Hold'em Personality Classification System

A poker simulation framework for command-and-control research. LLM-driven agents
with distinct personality types play automated Leduc Hold'em tournaments. Structured
observations of every decision are fed into a Linear Discriminant Analysis (LDA)
classifier that learns to identify personality type from gameplay behaviour alone —
without ever seeing the agent's instruction set.

---

## Overview

```
personalities (4) × tournaments (100) × hands (5) × slots (4)
        ↓
  observation vectors (19 features each)
        ↓
  LDA classifier  →  predicted personality
```

The four adversary personalities are: **Analytical**, **Conservative**, **Aggressive**, **Reckless**.

Each personality is driven by a separate instruction-set text file loaded at runtime.

---

## Folder Structure

```
leduc_holdem/
  config.yaml                  # All runtime configuration (no hardcoded values)
  requirements.txt             # Python dependencies
  main.py                      # CLI entry point

  game/
    deck.py                    # Card, Deck, hand_strength
    state.py                   # GameState dataclass + legal-action logic
    leduc_holdem.py            # Game loop, action resolution, showdown

  agents/
    base_agent.py              # Abstract BaseAgent
    personality_agent.py       # OpenAI-powered LLM agent (stateless per action)
    random_agent.py            # Uniform random over legal actions

  training/
    win_probability.py         # Exact win probability via full card enumeration
    observer.py                # 19-element observation vectors, CSV writer
    runner.py                  # ProcessPoolExecutor (4 workers), NDArray builder

  eval_runs/
    lda_pipeline.py            # PCA pre-processing + 7 LDA parameter sets
    model_selector.py          # Best model selection and pickle persistence

  instructions/                # Personality instruction-set .txt files (loaded at runtime)
  data/                        # Auto-created: observations/ and models/ (git-ignored)
```

---

## Game Rules (Leduc Hold'em)

| Parameter | Value |
|---|---|
| Deck | 6 cards: J♥ Q♥ K♥ J♠ Q♠ K♠ |
| Rank order | J < Q < K |
| Starting chips | $50 per player per tournament |
| Ante | $1 per player per hand |
| Hands per tournament | 5 |
| Pre-flop raise size / cap | $2 / 2 raises total |
| Post-flop raise size / cap | $4 / 2 raises total |
| Winning hand | Pair beats high card; higher rank wins ties |

A **pair** is formed when a player's private card matches the community card rank.
Ties split the pot (rounded down; remainder stays out).

---

## Observation Vector (19 features)

| Index | Feature | Encoding |
|---|---|---|
| 0 | Win probability | Float 0.0–1.0 |
| 1 | Personality agent stack | Raw chip count |
| 2 | Opponent stack | Raw chip count |
| 3 | Pot size | Raw chip count |
| 4–6 | Last opponent action | One-hot: [Check/Call, Raise, Fold] |
| 7–12 | Private card | One-hot: [J♥, Q♥, K♥, J♠, Q♠, K♠] |
| 13–18 | Community card | One-hot (all zeros pre-flop) |

Each hand produces exactly **4 observation slots** (2 pre-flop + 2 post-flop).
Unused slots are padded with `-1.0`. Each tournament produces a **20 × 19** CSV.

---

## LDA Pipeline

1. Load 100 CSVs per personality → stack to **(100, 380)** NDArray
2. 75/25 train/test split per personality
3. Concatenate → **X_train (300, 380)**, **X_test (100, 380)**
4. PCA: `n_components = min(n_samples - 1, n_features)`
5. Evaluate 7 LDA configurations (SVD, LSQR, Eigen × shrinkage values)
6. Select best by highest CV accuracy + lowest test mislabeling %
7. Save PCA + LDA to `data/models/best_lda_model.pkl`

---

## Setup

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt

# Set your OpenAI API key
$env:OPENAI_API_KEY = "sk-..."   # Windows PowerShell
# export OPENAI_API_KEY=sk-...   # macOS/Linux
```

---

## Usage

```bash
# Smoke test — 3 tournaments per personality (~12 API calls total)
python main.py --first-pass

# Full run — 100 tournaments per personality
python main.py

# Re-run LDA only (skip data collection, reuse existing CSVs)
python main.py --eval-only

# Use a custom config file
python main.py --config path/to/config.yaml
```

---

## Parallelization

Tournaments run in **4 parallel worker processes** (one per personality) via
`concurrent.futures.ProcessPoolExecutor`. Within each worker, tournaments execute
strictly sequentially to respect OpenAI API rate limits.

Seeds are fully deterministic: `seed = random_seed_base + tournament_index`.
Every seed is logged to `data/observations/seeds.log`.

---

## Configuration (`config.yaml`)

All parameters are defined in `config.yaml`. Nothing is hardcoded in the codebase.
Key sections:

```yaml
api:
  model: gpt-5.4-mini
  key_env_var: OPENAI_API_KEY

training:
  num_tournaments: 100
  first_pass_tournaments: 3
  n_workers: 4
  random_seed_base: 42

lda:
  kfold_splits: 4
```

---

## Requirements

- Python 3.10+
- `openai >= 1.30.0`
- `numpy >= 1.26.0`
- `scikit-learn >= 1.4.0`
- `pyyaml >= 6.0`
- `jinja2 >= 3.1.0`
- `json-repair >= 0.1.0`
- `tqdm >= 4.66.0`
