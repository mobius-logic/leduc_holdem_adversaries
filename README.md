# Leduc Hold'em Personality Classification System

A poker simulation framework for command-and-control research. Rule-based agents
with distinct personality types play automated Leduc Hold'em tournaments. Structured
observations of every decision are fed into a family of Linear Discriminant Analysis
(LDA) classifiers — one per action step — that learn to identify personality type
from gameplay behaviour accumulated in real time.

> **Current branch: `automation`**
> The active pipeline uses fully deterministic rule-based agents — no LLM or API
> key required. The original LLM-backed code is preserved in `leduc_holdem/archive/`
> for reference.

---

## Overview

```
personalities (4) × tournaments (N) × hands (15) × slots (4)
        ↓
  observation vectors (28 features each)  +  15 aggregate features
        ↓
  60 sequential LDA models (LDA_1 … LDA_60)
        ↓
  per-step personality probability estimates  →  identified personality
```

The four adversary personalities are: **Analytical**, **Conservative**, **Aggressive**, **Reckless**.

Each personality is implemented as explicit Python rule logic in `non_instruction/agents/`.
No instruction-set files or API calls are needed.

---

## Folder Structure

```
leduc_holdem/
  config.yaml                        # All runtime configuration (no hardcoded values)
  requirements.txt                   # Python dependencies

  non_instruction/                   # ★ Active pipeline — rule-based agents, no LLM
    main_ni.py                       #   CLI entry point (data generation + single LDA eval)
    runner_ni.py                     #   Tournament runner (ProcessPoolExecutor, 4 workers)
    server_ni.py                     #   Interactive Flask web server (port 5001)
    build_sequential_lda.py          #   Build LDA_1–LDA_60 step-indexed models
    agents/
      aggressive_rule_agent.py       #   Aggressive personality (bluff-heavy, raise to cap)
      analytical_rule_agent.py       #   Analytical personality (EV-pure, no bluffs)
      conservative_rule_agent.py     #   Conservative personality (tight, reactive)
      reckless_rule_agent.py         #   Reckless personality (never checks, desperation mode)

  game/
    deck.py                          # Card, Deck, hand_strength
    state.py                         # GameState dataclass + legal-action logic
    leduc_holdem.py                  # Game loop, action resolution, showdown

  agents/
    base_agent.py                    # Abstract BaseAgent
    random_agent.py                  # Uniform random over legal actions (used as opponent)

  training/
    win_probability.py               # Exact win probability via full card enumeration
    observer.py                      # 28-element observation vectors, CSV writer
    tournament_logger.py             # Per-tournament JSON log writer

  eval_runs/
    lda_pipeline.py                  # PCA pre-processing + 7 LDA parameter sets
    model_selector.py                # Best model selection and pickle persistence

  web_ni/
    index.html                       # Browser UI for interactive play vs rule-based agent

  data/                              # Auto-created: observations/ tournament/ models/ (git-ignored)
  LDA_models/                        # Sequential LDA_1–LDA_60 model files (git-ignored)
  archive/                           # LLM-based original code (git-ignored, kept for reference)
```

---

## Game Rules (Leduc Hold'em)

| Parameter | Value |
|---|---|
| Deck | 6 cards: J♥ Q♥ K♥ J♠ Q♠ K♠ |
| Rank order | J < Q < K |
| Starting chips | $25 per player per tournament |
| Ante | $1 per player per hand |
| Hands per tournament | 15 (configurable) |
| Pre-flop raise size / cap | $2 / 2 raises total |
| Post-flop raise size / cap | $4 / 2 raises total |
| Winning hand | Pair beats high card; higher rank wins ties |

A **pair** is formed when a player's private card matches the community card rank.
Ties split the pot (rounded down; remainder stays out).

---

## Observation Vector (28 features)

| Index | Feature | Encoding |
|---|---|---|
| 0 | Win probability | Float 0.0–1.0 |
| 1 | Personality agent stack | Raw chip count |
| 2 | Opponent stack | Raw chip count |
| 3 | Pot size | Raw chip count |
| 4–6 | Last opponent action | One-hot: [Check/Call, Raise, Fold] |
| 7–12 | Private card | One-hot: [J♥, Q♥, K♥, J♠, Q♠, K♠] |
| 13–18 | Community card | One-hot (all zeros pre-flop) |
| 19–21 | Last personality action | One-hot: [Check/Call, Raise, Fold] |
| 22 | `preflop_raise_Q` | Running Queen pre-flop raise rate (history so far) |
| 23 | `preflop_raise_K` | Running King pre-flop raise rate |
| 24 | `preflop_fold_J` | Running Jack pre-flop fold rate |
| 25 | `preflop_fold_Q` | Running Queen pre-flop fold rate |
| 26 | `postflop_raise_J` | Running J-high post-flop raise rate (no pair) |
| 27 | `postflop_fold_J` | Running J-high post-flop fold rate (no pair) |

Indices 22–27 are **running card-conditioned rates**: they reflect all personality
actions taken prior to the current one in the tournament, giving the LDA instant
intra-tournament behavioural history without waiting for the tournament to end.

Each hand produces exactly **4 observation slots** (up to 2 pre-flop + 2 post-flop).
Unused slots are padded with `-1.0`. Each tournament produces a **(15 × 4) × 28 = 60 × 28** CSV.

---

## Tournament-Level Aggregate Features (15 features)

Appended to the flattened observation matrix after each tournament to give LDA a
richer, personality-discriminating signal:

| Index | Feature | Description |
|---|---|---|
| 0 | `raise_rate` | Raise / total personality actions |
| 1 | `call_check_rate` | (Call + Check) / total actions |
| 2 | `fold_rate` | Fold / total actions |
| 3 | `net_winnings_norm` | `(final_stack − starting_chips) / starting_chips` |
| 4 | `hand_win_rate` | Weighted wins / total hands (tie = 0.5) |
| 5 | `preflop_raise_J` | Raise rate pre-flop with a Jack |
| 6 | `preflop_raise_Q` | Raise rate pre-flop with a Queen |
| 7 | `preflop_raise_K` | Raise rate pre-flop with a King |
| 8 | `preflop_fold_J` | Fold rate pre-flop with a Jack |
| 9 | `preflop_fold_Q` | Fold rate pre-flop with a Queen |
| 10 | `postflop_raise_pair` | Raise rate post-flop with a pair |
| 11 | `postflop_raise_K` | Raise rate post-flop with K-high (no pair) |
| 12 | `postflop_raise_Q` | Raise rate post-flop with Q-high (no pair) |
| 13 | `postflop_raise_J` | Raise rate post-flop with J-high (no pair) |
| 14 | `postflop_fold_J` | Fold rate post-flop with J-high (no pair) |

Saved as a companion `run_{seed}_{personality}_agg.csv` alongside each observation CSV.
Loaded and concatenated at evaluation time: **1680 obs + 15 agg = 1695 total features** per tournament.

---

## Sequential LDA Models (LDA_1 – LDA_60)

The system trains **60 step-indexed LDA models** rather than one global model.
`LDA_k` is trained on data that has exactly `k` observation slots filled (the rest
padded with `-1.0`), mirroring the partial-information state seen during live play.

**Feature vector for `LDA_k`:**
```
[  k × 28 obs features  ] + [ 15 agg features ]  =  k×28 + 15 total features
```

At inference time (interactive game), after each opponent action the step counter
increments and `LDA_step` is loaded to produce a fresh probability estimate over
the four personalities using all observations accumulated so far.

**Probability estimation detail:** LDA's raw `decision_function` scores can reach
~10⁷ magnitude on partially-padded inputs, collapsing `predict_proba` to 1.0/0.0.
The server z-scores the decision scores before applying a stable softmax, keeping
all four personality probabilities meaningful at every step.

---

## Global LDA Pipeline (optional, single model)

The original single-model pipeline is still available via `main_ni.py`:

1. Load N CSVs per personality → stack to **(N, 1695)** NDArray (1680 obs + 15 agg)
2. 75/25 train/test split per personality
3. Concatenate → **X_train**, **X_test**
4. PCA: `n_components = min(n_samples - 1, n_features)`
5. Evaluate 7 LDA configurations (SVD, LSQR, Eigen × shrinkage values)
6. Select best by highest CV accuracy + lowest test mislabeling %
7. Save PCA + LDA to `data/models/best_lda_model.pkl`

**CV fold behaviour:**
- `--first-pass`: uses **3-fold CV** (matches the training samples available at 20 tournaments × 0.75 split)
- Full run: uses **`lda.kfold_splits`** from `config.yaml` (default: **5**)
- Defensive cap: fold count is always clamped to the smallest class size with a logged warning

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

## Step 1 — Smoke Test (20 tournaments per personality, 3-fold CV)

Verify the full pipeline end-to-end before running the full dataset.
The tournament count for `--first-pass` is controlled by
`training.first_pass_tournaments` in `config.yaml` (default: **20**).
LDA automatically uses **3-fold CV** in this mode.

```powershell
cd leduc_holdem
.venv\Scripts\python.exe non_instruction/main_ni.py --first-pass
```

Expected output:
- 4 personalities × 20 tournaments each → 80 CSVs written to `data/observations/`
- 80 tournament JSON logs written to `data/tournament/`
- LDA evaluated with 3-fold CV, best model saved to `data/models/best_lda_model.pkl`

---

## Step 2 — Full Training Run (300 tournaments per personality, 5-fold CV)

Controlled by `training.num_tournaments` in `config.yaml` (default: **300**).
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

Use this to re-evaluate the classifier against existing CSVs without rerunning
tournaments, or to experiment with different `lda.kfold_splits` values in `config.yaml`.

```powershell
# Re-run LDA on full-run CSVs (5-fold CV)
cd leduc_holdem
.venv\Scripts\python.exe non_instruction/main_ni.py --eval-only

# Re-run LDA on first-pass CSVs (3-fold CV)
.venv\Scripts\python.exe non_instruction/main_ni.py --eval-only --first-pass
```

---

## Step 4 — Build Sequential LDA Models (LDA_1 – LDA_60)

After generating tournament data (Steps 1–2), build the 60 step-indexed models
used by the interactive game's real-time personality panel.

```powershell
cd leduc_holdem

# Build all 60 models (requires full training data in data/observations/)
.venv\Scripts\python.exe non_instruction/build_sequential_lda.py --start 1 --end 60

# Build a subset (e.g. only steps 24–60 if 1–23 already exist)
.venv\Scripts\python.exe non_instruction/build_sequential_lda.py --start 24 --end 60
```

Models are saved to `../LDA_models/LDA_{k}.pkl` (relative to `leduc_holdem/`).
This directory is git-ignored — models must be rebuilt locally after cloning.

Each model file contains a dict `{"pca": PCA, "lda": LinearDiscriminantAnalysis}`.
`LDA_k` is trained on feature vectors of size `k × 28 + 15`.

---

## Step 5 — Interactive Play (browser UI)

Play against any of the four rule-engine personalities in a browser. The LDA panel
shows real-time personality probability estimates updated after every opponent action.

```powershell
cd leduc_holdem
.venv\Scripts\python.exe non_instruction/server_ni.py
# Open http://localhost:5001
```

**Personality selection:**
- Choose **Analytical**, **Conservative**, **Aggressive**, or **Reckless** from the setup screen.
- Leave blank (or choose **Random**) to have the server pick a personality at random.
  In random mode the opponent is labelled "Opponent" and the personality is hidden.
  A **🔍 Reveal Opponent** button appears in the header — clicking it discloses the
  actual personality for that session.

**LDA panel (right sidebar):**
- Displays a live 4-row grid (Ana / Con / Agg / Rec) with probability bars.
- Updates after each opponent action, using the step-indexed `LDA_step` model that
  matches the number of actions seen so far.
- Requires `LDA_models/` to be populated (Step 4). If a model file is missing the
  panel entry for that step is silently skipped.

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
  hands_per_tournament: 15
  starting_chips: 25
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
  num_tournaments: 300          # Step 2 — full run
  first_pass_tournaments: 20    # Step 1 — smoke test
  n_workers: 4                  # parallel processes (one per personality)
  random_seed_base: 42

observation:
  vector_length: 28             # features per observation slot
  slots_per_hand: 4
  hands_per_tournament: 15
  pad_value: -1.0
  flat_length: 1695             # 15 hands × 4 slots × 28 features + 15 agg
  agg_length: 15

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
| **Reckless** | Always raises; J bluff-raises first pre-flop only | Pair → raise; K/Q high → raise; J high → fold | Yes | Desperation mode < 15 chips |

---

## Requirements

- Python 3.10+
- `flask >= 3.0`
- `numpy >= 1.26.0`
- `scikit-learn >= 1.4.0`
- `pyyaml >= 6.0`
