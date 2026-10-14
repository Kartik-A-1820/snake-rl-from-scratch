---
title: "02 · Deep Q-Networks on Snake (CNN on the board, CUDA)"
subtitle: "Function approximation, experience replay, target networks, Double DQN, and what the GPU is actually for"
---

# Summary

Tryout 01 showed that the algorithm wasn't the bottleneck; the observation was. Tryout 02 keeps Q-learning and changes two things:

1. **The state is the whole board.** It is a 4-channel 14×14 image: body recency, head, food, and the wall ring. The agent can finally see where its body is.
2. **The table becomes a neural network.** It is a small CNN with about 0.32 M parameters, trained with the two stabilisers from Mnih et al. (2015): an experience replay buffer and a target network. It also uses Double-DQN targets (van Hasselt et al., 2016).

The environment runs as 128 games stepped together as tensors on the GPU, and the replay buffer lives there too. The benchmark section measures where the GPU helps on this machine and where the CPU wins.

All numbers in this note come from `results.json`, which `python/train_dqn.py` writes (these runs: a laptop with a GTX 1650 Ti). If a figure below is empty, the run hasn't been done yet.

```widget
summary
```

# 1. From a table to a function

A table stores one number per (state, action). With the board as the state, that is hopeless: there are astronomically many board configurations. **Function approximation** replaces the table with $Q_\theta(s,a)$, a network with parameters $\theta$, and the network generalises across states that look alike.

Semi-gradient Q-learning minimises the squared TD error with the target held fixed:

$$
L(\theta) = \mathbb{E}_{(s,a,r,s')}\Big[\big(\underbrace{r + \gamma \max_{a'} Q_{\bar\theta}(s',a')}_{y,\ \text{treated as a constant}} - Q_\theta(s,a)\big)^2\Big],
\qquad
\nabla_\theta L = -2\,\mathbb{E}\big[(y - Q_\theta(s,a))\,\nabla_\theta Q_\theta(s,a)\big].
$$

This is *not* the gradient of any fixed objective. The target $y$ depends on $\theta$ too, and we ignore that dependence, hence "semi-gradient" (Sutton & Barto, 2018, §9.3). With a table, the update reduces exactly to tryout 01's rule, where $\nabla_\theta Q$ is a one-hot vector.

**The deadly triad** (Sutton & Barto, 2018, §11.3). Combining

1. function approximation,
2. bootstrapping, and
3. off-policy data

can diverge even on tiny problems (Baird, 1995; Tsitsiklis & Van Roy, 1997). DQN has all three. Its engineering choices are best read as ways to keep the triad in check.

# 2. Experience replay

Consecutive Snake frames are highly correlated, and SGD assumes roughly i.i.d. samples. **Replay** (Lin, 1992) stores transitions $(s, a, r, s', \text{terminal})$ in a ring buffer and trains on uniform random minibatches from it. This does three things:

- breaks temporal correlation;
- reuses each transition many times (better sample efficiency);
- averages the behaviour policy over many past policies, which damps feedback loops.

**Replay ratio.** Each vectorised step adds 128 transitions and runs one gradient step on a batch of 512, so each transition is sampled about $512/128 = 4$ times on average while it is in the buffer.

**What's stored.** The *pre-reset* next state, `info["next_obs"]`, and `terminal`. The `truncated` flag is deliberately not stored, so timeouts still bootstrap (tryout 01, §1).

**Memory, measured rather than guessed.** Observations are stored as `uint8`. Every value in the grid is $k/255$-exact for snakes up to 255 cells. 200 k transitions × 2 observations × 784 bytes ≈ **314 MB**. That is the largest thing on the GPU; the network weights are about **1.3 MB**. The "512 MB–1 GB VRAM budget" is therefore a replay-buffer budget, not a model budget.

# 3. Target network

If $y$ uses the same $\theta$ we are updating, every step moves the target as well, and errors can chase themselves. DQN computes $y$ with a *frozen copy* $\bar\theta$, synced every 1,000 gradient steps:

$$
y = r + \gamma\,(1 - \text{terminal})\, \max_{a'} Q_{\bar\theta}(s', a').
$$

Between syncs this turns learning into a sequence of supervised regression problems against a fixed target. That is the fitted-Q-iteration view of DQN (Riedmiller, 2005).

# 4. Overestimation and Double DQN

Even if every estimate is unbiased, $Q(s',a') = q(s',a') + \epsilon_{a'}$ with $\mathbb{E}[\epsilon]=0$, the max is biased upward by Jensen's inequality:

$$
\mathbb{E}\big[\max_{a'} Q(s',a')\big] \;\ge\; \max_{a'} \mathbb{E}\big[Q(s',a')\big] = \max_{a'} q(s',a').
$$

For $m$ actions with i.i.d. noise $\mathcal{N}(0,\sigma^2)$ and equal true values, the bias is $\sigma\,\mathbb{E}[\max_i Z_i]$: about $0.85\sigma$ for $m=3$. Bootstrapping feeds that bias back into the next target.

**Double Q-learning** (van Hasselt, 2010) decouples *selecting* the action from *evaluating* it. Double DQN (van Hasselt, Guez & Silver, 2016) uses the networks DQN already has:

$$
y^{\text{Double}} = r + \gamma (1-\text{terminal})\, Q_{\bar\theta}\big(s',\ \arg\max_{a'} Q_\theta(s',a')\big).
$$

When the selection noise is independent of the evaluation noise, $\mathbb{E}[Q_{\bar\theta}(s', \hat a)] = q(s', \hat a) \le \max_a q(s',a)$. The bias is gone, and slight underestimation is the price.

```widget
overestimation
```

**Dueling heads** (Wang et al., 2016) are available with `--dueling`. They split the head into $V(s)$ and $A(s,a)$, combined as $Q = V + A - \bar A$. This helps when many states have actions of nearly equal value, which in Snake is most of open space. Tryout 03 is a good place for the ablation.

# 5. A state that can see the body

| Channel | Content | Why |
|---|---|---|
| 0 | body recency: head $=1$, …, tail $=1/L$ | shape *and* direction of motion; which cells free up soonest |
| 1 | head one-hot | makes the egocentric reference explicit |
| 2 | food one-hot | the goal |
| 3 | wall ring (padding = 1) | lets convolutions see the boundary as "body-like danger" |

The network: Conv(4→32, 3×3) → Conv(32→64, 3×3, stride 2) → Conv(64→64, 3×3, stride 2), each followed by ReLU, so the 14×14 input shrinks to 7×7 and then 4×4. Then a dense layer from 1024 to 256 with ReLU, and a final dense layer to 3 Q-values.

- **Parameters:** 319,779, of which 82% sit in the 1024→256 dense layer.
- **Receptive field:** after three 3×3 convolutions with two of stride 2, each unit sees a 9×9 window. The dense layer then combines all 4×4 positions, so the Q-values depend on the whole board.
- **Tied to board size:** the dense layer fixes the input to 12×12. A board-size-agnostic variant would end with global pooling, which is a deliberate trade-off.

**Ablation:** `--obs basic` trains the *same* DQN on tryout 01's 11 features with a 5 K-parameter MLP. If the thesis is right, it plateaus near 20, like the table. If the CNN clearly beats about 20, the improvement is attributable to the state, not to "deep RL".

# 6. Vectorised environments on the GPU, and what the GPU is actually for

`snake_rl/vec_env.py` steps N games as tensors:

- The body is an *age grid*: each body cell stores how many ticks it remains occupied. Moving decrements every cell and writes the new head; eating skips the decrement.
- Food placement reproduces the scalar engine's RNG bit for bit: mulberry32 in int64, with the "k-th free cell" found by a cumulative sum.
- `tests/test_parity.py` checks it step by step against the Python engine. That engine is in turn checked against traces from the JavaScript game.

Why bother? A small CNN at batch 1 is dominated by kernel-launch and synchronisation overhead on a GPU (tens of µs per call). The CPU often wins there. The GPU only pays off when each call carries enough work: many environments per step, and large training batches. The benchmark makes that concrete on *this* laptop.

```widget
bench
```

**How to read it.**

- Compare the CUDA and CPU rows at batch 1 (the live game) and at batch 512–4096 (training).
- For the 5 K-parameter MLP, expect the CPU to win at small batches.
- For the CNN, expect CUDA to win from a few hundred samples per batch.
- Environment throughput on CUDA should grow almost linearly with N until the GPU saturates.

# 7. How the theory maps to the code

| Theory | Code (`python/train_dqn.py`) |
|---|---|
| ε-greedy behaviour, ε 1 → 0.02 over 40% of steps | `explore = torch.rand(n) < eps; act = torch.where(explore, random, greedy)` |
| replay buffer (Lin, 1992) | `Replay.add/sample`, uint8 on the training device |
| target $y$ with frozen $\bar\theta$ | `target(o2)`, `target.load_state_dict(net.state_dict())` every 1,000 updates |
| Double DQN | `a2 = net(o2).argmax(1); q2 = target(o2).gather(1, a2)` |
| terminal vs truncated | `y = r + γ(1 − term)·q2`; `trunc` is not stored as terminal |
| Huber loss (Mnih et al., 2015) | `F.smooth_l1_loss(q, y)`, gradients clipped at norm 10 |
| evaluation protocol | greedy on the same 500 *test* seeds as tryout 01; "best" chosen on 200 separate *validation* seeds |

# 8. Results

```widget
curves
```

```widget
checkpoint_table
```

```widget
loss_q
```

**Analysis.** The run shown above is the repaired setup trained for 5 M steps (`dqn_cnn_12x12_long`; the first, failed run is §8b). The four questions, in order:

1. **Does the CNN beat the tabular agent and the greedy heuristic (both about 20)?** Marginally, and not reliably. The best checkpoint by validation score (epoch 65, validation 24.6) has test mean **24.5**, median 26, p90 37, max 45. But the last eight checkpoints on the test seeds read 24.5, 21.3, 19.0, 21.6, 22.3, 18.5, 16.9, 24.9: mean 21.1, spread about ±2.7. The best-checkpoint number is a maximum over 20 noisy evaluations, so it is optimistic; the honest summary is "about 21, roughly level with tabular and the MLP (20.9)", on one training seed. The spread between three seeds at 1 M steps (§8b: 14.3 to 18.2) is of the same size as the gap to 20.
2. **Did the cause of death shift away from self-collision?** No, it grew. At the best checkpoint 443 of 500 games end by self-collision, 39 by wall and 18 by timeout; timeouts fell from 357 of 500 at 0.5 M steps to under 50. The agent stopped circling and walling itself in early but still traps itself once the body is long. Winning (filling all 144 cells) never happened; the best fill reached 33–36% of the board.
3. **Did Q-values blow up?** No. The mean Q of the sampled action rose from about 0 to 6 by epoch 80 and then stayed flat at 6.0–6.3, while the Huber loss settled near 0.1. No sign of divergence or of the deadly triad at this scale. (Q is a discounted value, so it is not directly comparable to the food count.)
4. **Is the best checkpoint the last one?** No. The best by validation is epoch 65 of 100; the final epoch 100 scored 24.9 on test but 23.4 on validation, and epochs 90 and 95 dipped to 18.5 and 16.9. With a constant learning rate and no target-network or learning-rate decay, the policy keeps wandering at roughly the same level between checkpoints. A learning-rate schedule, or averaging evaluations over several checkpoints, would make the late-training curve smoother; neither is implemented.

# 8b. The first CNN run failed: diagnosis, and reward shaping

The first 1 M-step run (CPU and, identically, CUDA) did **not** learn: mean food 0.15 → 0.9, loss ≈ 0.002, $Q \approx 0$, against about 20 for the tabular agent. A failed run is useful data, so it is kept here.

## What the logs said

| Evidence | Reading |
|---|---|
| CPU and CUDA runs show the same curve, 7 656 gradient updates in 1 M env steps | Not a device bug. Same algorithm, same failure. |
| Greedy evaluation: 500 of 500 games end by `timeout` | The greedy policy circles until the starvation limit. Circling is safe, and nothing in the reward punishes it. |
| 7 656 updates ÷ 1 000 (`--target-every`) = **7 target-network syncs** | Bootstrapping advanced about 7 steps of horizon in total. The $+1$ for food could not propagate back more than a few steps. |
| Same code, `--obs basic` (11-feature MLP): test mean **20.9** at epoch 10 | The replay, target network, Double-DQN and loss are correct. The problem is specific to the CNN-on-grid run. |
| CNN with 4× the updates (batch 128, 4 updates per step, target sync every 500) and 3-step returns: test mean 1.9 → 4.4 → 11.0 → **16.8**, timeouts 265 → 52 of 500 | The main cause was too few optimiser steps and too few target syncs, not the observation. |

So hypotheses 1 (too few updates) and 4 (target-network schedule) were confirmed, and 2 (the circling local optimum) was confirmed as the symptom. The weaker suspicion that the grid hides the heading (actions are relative turns and the grid has no direction plane) is **not** the main cause. It may still explain why the CNN lags the MLP.

Two implementation facts matter for any change to the reward. A starvation `timeout` is a *truncation*: the stored transition has `disc = γᵏ`, so it still bootstraps from the next state. Only a death or a win is a terminal and sets the bootstrap to 0. And the stored next state is the one *before* auto-reset.

## Potential-based shaping

Define a potential on states, $\Phi(s) = -k\,\frac{|x_h - x_f| + |y_h - y_f|}{2S}$ (Manhattan distance to the food, scaled so $\Phi \in [-k, 0]$), and add

$$F(s, s') = \gamma\,\Phi(s') - \Phi(s)$$

to the reward. A move toward the food gives $F > 0$, a move away gives $F < 0$. This is the "reward moving closer, hurt moving away" idea, in the one form that is safe.

**Theorem (Ng, Harada & Russell, 1999).** For any $\Phi$, the shaped MDP has the same optimal policies as the original one. Sketch: along any trajectory $s_0, s_1, \dots, s_T$ the shaping terms telescope,

$$\sum_{t=0}^{T-1}\gamma^t F(s_t, s_{t+1}) = \sum_{t=0}^{T-1}\gamma^t\bigl(\gamma\Phi(s_{t+1}) - \Phi(s_t)\bigr) = \gamma^T\Phi(s_T) - \Phi(s_0).$$

Taking the expectation of the shaped return from $(s, a)$ gives $Q'(s,a) = Q(s,a) - \Phi(s)$, because $\Phi(s_0)$ does not depend on the action and $\gamma^T\Phi(s_T) \to 0$ (or is 0 by choice at a terminal). Subtracting a function of $s$ alone from every action's value leaves the $\arg\max_a$ unchanged. $\blacksquare$

The conditions that make the proof go through are exactly the implementation details:

- $\Phi(\text{terminal}) = 0$ for death and for a win. Otherwise the telescoping leaves a $\gamma^T\Phi(s_T)$ that depends on where the snake died and biases the policy.
- After eating, $s'$ contains the **new** food, so $\Phi(s')$ is measured to the new food. The snake is not rewarded for "arriving" at food that has just vanished.
- A timeout is not a terminal. $s'$ is a real state, so $\Phi(s')$ is kept and bootstrapping continues.
- $\gamma$ in $F$ must be the training $\gamma$.

`python/tests/test_env.py` checks the telescoping identity on a real trajectory ($\sum\gamma^tF = -\Phi(s_0)$ when the episode ends in death) and that the torch vec env and the scalar engine give the same shaped rewards.

## Why the naive version is not safe

"+$c$ if closer, $-c$ if farther" is not of the form $\gamma\Phi(s') - \Phi(s)$, so none of the above applies. The sign of the reward depends only on the distance to the food, never on whether the move is wise:

- **Bias.** A long snake often *must* move away from the food to avoid sealing itself in. Naive shaping charges a fee for every such detour, so the optimum of the shaped problem is a more reckless snake than the true optimum. With $\gamma\Phi$ the same detour is repaid when the snake comes back, because the telescoped total is path-independent.
- **Exploitable loops.** A reward that depends on the sign of a distance change, with no discount-consistent offset, can pay differently on the approach and on the retreat of the same cycle (for example when the food moves or the path is asymmetric). The agent will find whichever cycle pays most.
- **Scale.** $\pm 1$ per move would dominate the food reward ($+1$) over a ten-step approach. The runs below use $c = 0.1$, which is already large next to the potential's per-step change of about $k/(2S) = 0.04$.

## The timeout loophole is a different fix

Potential shaping preserves the optimal policy, so it **cannot** by itself remove an optimum that the original reward already allows: with a reward of 0 per step and $-1$ for death, circling forever has value 0 and is a legitimate local optimum. Closing that loophole means deliberately changing the task. The fourth arm adds a reward of $-1$ on a starvation timeout (`--timeout-penalty 1`). That is a change of problem, not a shaping term, and the notes keep the two ideas separate.

## Ablation

Four arms on CUDA, same seed, 1 M env steps, CNN on the grid, the same repaired training setup (32 envs, batch 128, 4 updates per env step, target sync every 500 updates, 3-step returns, Double DQN). Only the reward differs. Test mean is over the 500 held-out seeds, for the checkpoint with the best validation score.

| # | Reward | Test mean | Median | p90 | Max | Deaths (wall / self / timeout, of 500) |
|---|---|---|---|---|---|---|
| 1 | baseline ($+1$ food, $-1$ death) | **16.83** | 18 | 29 | **36** | 112 / 336 / 52 |
| 2 | naive $\pm 0.1$ per move | 12.99 | 14 | 19 | 24 | 48 / 397 / 55 |
| 3 | potential, $k = 1$ | 11.63 | 13 | 19 | 26 | 56 / 351 / 93 |
| 4 | potential, $k = 1$, timeout $-1$ | 12.26 | 14 | 20 | 25 | 73 / 329 / 98 |

Learning curves (test mean at epochs 5, 10, 15, 20, i.e. every 250 k steps):

| # | 0.25 M | 0.50 M | 0.75 M | 1.0 M |
|---|---|---|---|---|
| 1 baseline | 1.8 | 4.4 | 11.0 | 16.8 |
| 2 naive | 2.5 | 8.2 | 13.0 | 12.5 |
| 3 potential | 1.7 | 4.6 | 10.0 | 11.6 |
| 4 potential + timeout | 1.6 | 4.8 | 11.1 | 12.3 |

**Reading, with the caveats up front.** This is **one seed per arm**, and between-run spread in this kind of training is often several points, so differences of 1–4 points here are not established.

- Once the update schedule is repaired, **shaping did not help within 1 M steps**. The baseline is the best arm at the end.
- Naive shaping learned fastest early (8.2 at 0.5 M against 4.4), consistent with a denser signal, and then stalled and drifted down (13.0 → 12.5). Its deaths shifted from wall to self-collision (397 of 500), which fits the bias argument: the agent is paid to chase the food and does not plan around its own tail. One seed cannot prove that mechanism.
- Potential shaping did not speed up early learning in this run. The potential's per-step reward ($\le 0.04$) is small next to the noise in the Q-estimates at this stage, so its effect would show up later, if at all.
- The timeout penalty did not reduce timeouts (98 against 93 for arm 3). Timeouts were already falling in every arm as the policy improved (baseline: 265 → 52), so the loophole shrinks as a side effect of learning to eat, without the penalty.
- The larger lever by far was the update schedule, not the reward: the same reward went from a test mean under 1 to 16.8.

**More seeds, baseline against potential only** (same 1 M-step setup; test mean of the best checkpoint):

| Seed | 1 baseline | 3 potential, $k = 1$ |
|---|---|---|
| 0 | 16.83 | 11.63 |
| 1 | 18.23 | 12.16 |
| 2 | 14.30 | 11.37 |
| mean | **16.45** | 11.72 |

The baseline is ahead in all three seeds (by 5.2, 6.1 and 2.9 points), and the ranges do not overlap (14.3 to 18.2 against 11.4 to 12.2). Three seeds is still modest evidence, but it supports the reading that, here, potential shaping with $k = 1$ slowed learning within 1 M steps instead of speeding it up. That does not contradict the theory: shaping keeps the optimum, it does not promise faster learning, and a poorly scaled $\Phi$ adds noise to the targets.

Not run: other values of $k$, annealing $k \to 0$ (`--shape-anneal` is implemented), shaping on the 5 M-step run, and the naive and timeout arms on more seeds (those two rest on a single seed).

# 9. Watch it play

The replays below are re-simulated by the game engine from the seed and the recorded actions, so they are exact. The bars show the network's Q-values at every step.

```widget
replay
```

To run the network live in the game, export it first with `run.ps1 export`. Then open `game/index.html`, go to **Models → Choose file**, and pick `models/dqn_cnn_12x12_best.json`.

# 10. Failure modes to watch for

- **Divergence.** Q-values grow without bound and the loss spikes after a target sync. Remedies: lower the learning rate, sync the target less often, use Huber loss, clip gradients.
- **Catastrophic forgetting.** Once the agent is good, the buffer fills with long late-game episodes and early-game skills decay. Watch for test scores that rise and then fall.
- **Looping.** A cautious agent circles forever. Timeouts cut this off as truncation, which is correct, but a high timeout rate means the agent is valuing safety over food. Reward shaping or a smaller γ can trade this off.
- **Overfitting to seeds.** Training uses millions of distinct food sequences, so this is unlikely. That is still why test and validation seeds are disjoint.

# Self-check

```quiz
Q: Why does DQN compute the TD target with a separate, periodically copied network?
- [ ] To halve GPU memory
- [x] So the regression target does not move with every gradient step, which stabilises bootstrapping with function approximation
- [ ] Because the online network cannot evaluate s'
Why: With function approximation a single update changes Q everywhere, including at s'. A frozen target turns each sync interval into an ordinary regression problem (fitted Q-iteration).
```

```quiz
Q: Each Q estimate for 3 actions has zero-mean noise. Why is r + γ·max Q(s',·) biased upward?
- [ ] Because γ < 1
- [x] E[max of noisy estimates] ≥ max of their means (Jensen), and bootstrapping propagates the excess
- [ ] Because rewards are positive
Why: The max preferentially picks whichever estimate got lucky. Double DQN selects with one network and evaluates with the other, so the selection's luck is not reused in the evaluation.
```

```quiz
Q: On a GTX 1650 Ti, which is most likely faster for choosing ONE action with a 5 K-parameter MLP?
- [x] CPU (PyTorch with 1 thread, or ONNX Runtime)
- [ ] CUDA, because GPUs are always faster for neural networks
- [ ] It makes no difference
Why: At batch 1 the work is a few thousand multiply-adds. GPU time is dominated by kernel launches, synchronisation and host-device copies, so the CPU finishes before the GPU has started. The benchmark in §6 measures this on your laptop.
```

# References

- Baird, L. (1995). Residual algorithms: reinforcement learning with function approximation. *ICML*.
- Hessel, M., et al. (2018). Rainbow: combining improvements in deep reinforcement learning. *AAAI*.
- Lin, L.-J. (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching. *Machine Learning*, 8.
- Mnih, V., et al. (2013). Playing Atari with deep reinforcement learning. *arXiv:1312.5602*.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- Riedmiller, M. (2005). Neural fitted Q iteration. *ECML*.
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized experience replay. *ICLR*.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), §9.3, §11.3. MIT Press.
- Tsitsiklis, J. N., & Van Roy, B. (1997). An analysis of temporal-difference learning with function approximation. *IEEE Transactions on Automatic Control*, 42(5).
- van Hasselt, H. (2010). Double Q-learning. *NeurIPS*.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *AAAI*.
- Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. *ICML*.
