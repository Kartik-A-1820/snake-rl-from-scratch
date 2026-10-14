# Tryout 01: tabular Q-learning, checkpoint log

**Method:** one-step Q-learning (Watkins & Dayan 1992). A 2048×3 table is indexed by the 11 binary "basic" features: danger ahead/right/left, heading, and food direction.

**Run:** `node tools/train_tabular.js --size 12` (and `--size 6`). The run is deterministic, so the same arguments give the same numbers.
These numbers came from the same JS engine running in a browser, because the Python sandbox was down this session. Seed 1 on 12×12 reproduces the earlier preliminary run exactly (2,173,663 env steps).

| Setting | Value |
|---|---|
| Epoch | 100 training episodes |
| Run length | 200 epochs (20,000 episodes) |
| Checkpoints | every 25 epochs |
| α (learning rate) | 0.1 |
| γ (discount) | 0.95 |
| ε (exploration) | linear 1.0 → 0.01 over the first 100 epochs, then held at 0.01 |
| Rewards | +1 food, −1 death, 0 otherwise |
| Timeout | episode is cut (truncated, not terminal) after 100 × length steps without food |
| EVAL | greedy policy (ε = 0) on 500 held-out seeds (100000–100499), same seeds at every checkpoint |
| Train columns | scores while exploring, over the 25-epoch window before the checkpoint |

Training scores lag behind eval because ε-random moves kill the snake. **Judge progress by EVAL.**

## 12×12 board (win = score 141, i.e. fill all 144 cells)

| Epoch | ε | Train mean (max) | EVAL mean | median | p90 | max | Wins | Δ eval mean | Deaths wall / self / timeout |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0   | 1.00  | –            | 0.04  | 0  | 0  | 1  | 0 | –     | 100% / 0% / 0% |
| 25  | 0.753 | 0.41 (4)     | 6.67  | 6  | 13 | 23 | 0 | +6.63 | 28% / 51% / 21% |
| 50  | 0.505 | 1.24 (7)     | 12.51 | 12 | 19 | 28 | 0 | +5.84 | 8% / 92% / 0% |
| 75  | 0.258 | 2.68 (14)    | 18.09 | 18 | 27 | 40 | 0 | +5.58 | 8% / 92% / 0% |
| 100 | 0.010 | 7.80 (37)    | 20.03 | 19 | 29 | 49 | 0 | +1.93 | 8% / 92% / 0% |
| 125 | 0.010 | 17.45 (46)   | 19.16 | 19 | 28 | 40 | 0 | −0.86 | 9% / 91% / 0% |
| 150 | 0.010 | 17.74 (46)   | 19.75 | 20 | 29 | 41 | 0 | +0.59 | 6% / 94% / 0% |
| 175 | 0.010 | 17.95 (43)   | 21.68 | 21 | 31 | 42 | 0 | +1.93 | 8% / 92% / 0% |
| 200 | 0.010 | 18.17 (47)   | 20.08 | 20 | 28 | 39 | 0 | −1.61 | 12% / 88% / 0% |

The best single eval game reached 52 of 144 cells (36%), at epoch 100.

**Seed check:** eval mean at epochs 0 / 25 / 50 / … / 200 for seeds 2 and 3. Wins were 0 throughout.

| Seed | Eval mean by checkpoint |
|---|---|
| 2 | 0.04 → 8.24 → 11.45 → 18.18 → 21.21 → 19.73 → 19.21 → 19.24 → 21.25 |
| 3 | 0.04 → 8.96 → 12.14 → 19.07 → 18.97 → 20.00 → 19.91 → 19.03 → 20.13 |

## 6×6 board (win = score 33): added so "is it winning?" has a meaningful answer

| Epoch | ε | Train mean (max, wins) | EVAL mean | median | p90 | max | Wins / 500 | Δ eval mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0   | 1.00  | –             | 0.07  | 0  | 0  | 2  | 0 | –     |
| 25  | 0.753 | 0.39 (4, 0)   | 6.83  | 6  | 11 | 15 | 0 | +6.76 |
| 50  | 0.505 | 0.97 (6, 0)   | 8.50  | 8  | 13 | 18 | 0 | +1.67 |
| 75  | 0.258 | 2.27 (13, 0)  | 12.13 | 12 | 18 | 26 | 0 | +3.63 |
| 100 | 0.010 | 6.13 (25, 0)  | 13.75 | 14 | 19 | 28 | 0 | +1.62 |
| 125 | 0.010 | 11.69 (33, **2**) | 12.85 | 12 | 19 | 30 | 0 | −0.90 |
| 150 | 0.010 | 12.18 (28, 0) | 13.28 | 13 | 19 | 28 | 0 | +0.43 |
| 175 | 0.010 | 11.55 (33, **1**) | 12.75 | 13 | 18 | 24 | 0 | −0.53 |
| 200 | 0.010 | 11.85 (33, **1**) | 13.16 | 13 | 18 | 29 | 0 | +0.41 |

Seed 2 on 6×6 had **1 eval win out of 500** at epoch 100. Seed 3 never won.

## Reading the numbers

1. **Learning is real but short-lived.** On 12×12, eval mean goes 0 → 6.7 → 12.5 → 18.1 → 20.0 over the first 100 epochs. After that it stops improving. The ±1–2 swings between later checkpoints are noise (three seeds all end at 20–21), not learning or forgetting.
2. **It ties a hand-written rule.** The greedy heuristic (move toward food, avoid instant death) scores 20.4 on the same board. The Q-table agrees with that heuristic on 87.6% of states. It has learned roughly that rule and nothing more.
3. **Why it plateaus: it can't see its own body.** Only 256 of the 2,048 table rows can ever occur (8 danger combinations × 4 headings × 8 food directions). About 90% of deaths are self-collisions, where the snake coils into a pocket it can't escape. Nothing in the 11 features describes free space beyond the next cell. Many different board situations map to the same row, so Q-learning is optimising a heavily aliased (partially observable) problem.
4. **Is it winning?** On 12×12, no: 0 wins in 20,000 training episodes and 4,500 eval games. The best game filled 36% of the board. On 6×6, almost never: seed 1 won 4 of 20,000 training episodes (~0.02%) and 0 of its 4,500 eval games; across all three seeds there was 1 eval win in 13,500 games. Those wins happen when the food happens to land conveniently, not from planning.
5. **What fixes it:** state the network can actually use. Next steps, in order:
   1. "Extended" features: the reachable free area after each candidate move (already in the engine), fed to a small MLP DQN.
   2. The full grid fed to a small CNN.
   3. After that, the algorithm upgrades (Double/Dueling DQN, PPO).

   Changing the algorithm on the 11 features won't help: the information isn't in the state.

Raw data: `checkpoints_12x12_seed1.json` / `checkpoints_6x6_seed1.json` (written when you run the script locally) and `prelim_metrics.json`. The model is `models/tabular_q_12x12_seed1.json`: 256 states, sparse JSON, ~8 KB.
