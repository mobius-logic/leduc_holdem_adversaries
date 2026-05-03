# Leduc Hold'em Personality Classification System

A poker simulation framework for command-and-control research. Rule-based agents
with distinct personality types play automated Leduc Hold'em tournaments. Structured
observations of every decision are fed into a Linear Discriminant Analysis (LDA)
classifier that learns to identify personality type from gameplay behaviour alone.

> **Current branch: `automation`**
> The active pipeline uses fully deterministic rule-based agents — no LLM or API
> key required. The original LLM-backed code is preserved in `leduc_holdem/archive/`
> for reference.

---

## Overview

```
personalities (4) × tournaments (N) × hands (5) × slots (4)
        ↓
  observation vectors (19 features each)
        ↓
  LDA classifier  →  predicted personality
```

The four adversary personalities are: **Analytical**, **Conservative**, **Aggressive**, **Reckless**.

Each personality is implemented as explicit Python rule logic in `non_instruction/agents/`.
No instruction-set files or API calls are needed.

---

## Folder Structure

```
leduc_holdem/
  config.yaml                  # All runtime configuration (no hardcoded values)
  requirements.txt             # Python dependencies

  non_instruction/             # ★ Active pipeline — rule-based agents, no LLM
    main_ni.py                 #   CLI entry point (mirrors old main.py)
    runner_ni.py               #   Tournament runner (ProcessPoolExecutor, 4 workers)
    server_ni.py               #   Interactive Flask web server (port 5001)
    agents/
      aggressive_rule_agent.py #   Aggressive personality (bluff-heavy, raise to cap)
      analytical_rule_agent.py #   Analytical personality (EV-pure, no bluffs)
      conservative_rule_agent.py # Conservative personality (tight, reactive)
      reckless_rule_agent.py   #   Reckless personality (never checks, desperation mode)

  game/
    deck.py                    # Card, Deck, hand_strength
    state.py                   # GameState dataclass + legal-action logic
    leduc_holdem.py            # Game loop, action resolution, showdown

  agents/
    base_agent.py              # Abstract BaseAgent
    random_agent.py            # Uniform random over legal actions (used as opponent)

  training/
    win_probability.py         # Exact win probability via full card enumeration
    observer.py                # 19-element observation vectors, CSV writer
    tournament_logger.py       # Per-tournament JSON log writer

  eval_runs/
    lda_pipeline.py            # PCA pre-processing + 7 LDA parameter sets
    model_selector.py          # Best model selection and pickle persistence

  web_ni/
    index.html                 # Browser UI for interactive play vs rule-based agent

  data/                        # Auto-created: observations/ tournament/ models/ (git-ignored)
  archive/                     # LLM-based original code (git-ignored, kept for reference)
```

---

## Game Rules (Leduc Hold'em)

| Parameter | Value |
|---|---|
| Deck | 6 cards: J♥ Q♥ K♥ J♠ Q♠ K♠ |
| Rank order | J < Q < K |
| Starting chips | $50 per player per tournament |
| Ante | $1 per player per hand |
| Hands per tournament | 5 (configurable) |
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

1. Load N CSVs per personality → stack to **(N, 380)** NDArray
2. 75/25 train/test split per personality
3. Concatenate → **X_train**, **X_test**
4. PCA: `n_components = min(n_samples - 1, n_features)`
5. Evaluate 7 LDA configurations (SVD, LSQR, Eigen × shrinkage values)
6. Select best by highest CV accuracy + lowest test mislabeling %
7. Save PCA + LDA to `data/models/best_lda_model.pkl`

**CV fold behaviour:**
- `--first-pass`: uses **3-fold CV** (matches the ~3 training samples per class at 5 tournaments × 0.75 split)
- Full run: uses **`lda.kfold_splits`** from `config.yaml` (default: **5**)
- Defensive cap: fold count is always clamped to the smallest class size with a logged warning, preventing `StratifiedKFold` failures regardless of dataset size

---

## Setup

```powershell
cd leduc_holdem

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

No API key is required. All agents are pure Python rule engines.

---

## Step 1 — Smoke Test (5 tournaments per personality, 3-fold CV)

Verify the full pipeline end-to-end before running the full dataset.
The tournament count for `--first-pass` is controlled by
`training.first_pass_tournaments` in `config.yaml` (default: **5**).
LDA automatically uses **3-fold CV** in this mode.

```powershell
cd leduc_holdem
.venv\Scripts\python.exe non_instruction/main_ni.py --first-pass
```

Expected output:
- 4 personalities × 5 tournaments each → 20 CSVs written to `data/observations/`
- 20 tournament JSON logs written to `data/tournament/`
- LDA evaluated with 3-fold CV, best model saved to `data/models/best_lda_model.pkl`

---

## Step 2 — Full Training Run (100 tournaments per personality, 5-fold CV)

Controlled by `training.num_tournaments` in `config.yaml` (default: **100**).
LDA uses **5-fold CV** (`lda.kfold_splits` in config).

```powershell
cd leduc_holdem
.venv\Scripts\python.exe non_instruction/main_ni.py
```

Runs 4 personalities in parallel (one worker process each). Within each worker,
tournaments execute sequentially. Seeds are deterministic:
`seed = random_seed_base + tournament_index`. Every seed is logged to
`data/observations/seeds.log`.

---

## Step 3 — Re-run LDA Only (skip data collection)

Use this to re-evaluate the classifier against existing CSVs without rerunning tournaments,
or to experiment with different `lda.kfold_splits` values in `config.yaml`.

```powershell
# Re-run LDA on full-run CSVs (5-fold CV)
cd leduc_holdem
.venv\Scripts\python.exe non_instruction/main_ni.py --eval-only

# Re-run LDA on first-pass CSVs (3-fold CV)
.venv\Scripts\python.exe non_instruction/main_ni.py --eval-only --first-pass
```

---

## Step 4 — Interactive Play (browser UI)

Play against any personality's rule engine in a browser.

```powershell
cd leduc_holdem
.venv\Scripts\python.exe non_instruction/server_ni.py
# Open http://localhost:5001
```

Select a personality (Analytical / Conservative / Aggressive / Reckless) or leave
blank for a random pick. The agent responds instantly — no network latency.

To use a different port:

```powershell
.venv\Scripts\python.exe non_instruction/server_ni.py --port 8080
```

---

## Configuration (`config.yaml`)

All parameters are defined in `config.yaml`. Nothing is hardcoded in the codebase.
Key sections:

```yaml
game:
  hands_per_tournament: 5
  starting_chips: 50
  ante: 1
  preflop_raise_size: 2
  preflop_raise_cap: 2
  postflop_raise_size: 4
  postflop_raise_cap: 2

training:
  personalities:
    - analytical
    - conservative
    - aggressive
    - reckless
  num_tournaments: 100          # Step 2 — full run
  first_pass_tournaments: 5     # Step 1 — smoke test
  n_workers: 4                  # parallel processes (one per personality)
  random_seed_base: 42

lda:
  kfold_splits: 5               # full-run CV folds; --first-pass always uses 3
```

---

## Personality Rules Summary

| Personality | Pre-flop | Post-flop | Bluffs? | Stack management |
|---|---|---|---|---|
| **Aggressive** | Always raises K/Q; bluff-raises J | Pair → raise to cap; K high → semi-bluff; Q/J → bluff then fold | Yes | +10% confidence bonus |
| **Analytical** | K raises; Q/J check/fold | Pair → raise; K high → raise if win% > 60%; rest → fold | No | Folds earlier below 10 chips |
| **Conservative** | Never raises pre-flop; folds to raises | Pair → reactive raise only; K high → call if win% > 55%; rest → fold | No | −15% probability discount; lockdown < 20 chips |
| **Reckless** | Always raises; J bluff-raises | Pair → raise to cap; K/Q high → raise; J → bluff once then fold | Yes | Desperation mode < 15 chips |

---

## Requirements

- Python 3.10+
- `flask >= 3.0`
- `numpy >= 1.26.0`
- `scikit-learn >= 1.4.0`
- `pyyaml >= 6.0`
