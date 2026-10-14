---
title: "05 · The win challenge: can reinforcement learning fill the whole board?"
subtitle: "Curriculum learning, sparse rewards, long horizons, and a planner that never loses"
---

# Summary

Every agent so far scores around 20 on the 12×12 board. A *win* means filling all 144 cells (141 foods), and none of them has ever done it. This tryout asks whether a learned agent can win at all, how far up in board size it gets, and what separates it from a trivial planning algorithm that always wins.

**An honest framing first.** Winning Snake is a long-horizon planning problem. The reward for the final bite arrives hundreds of steps after the decisions that made it possible, and a policy that has never won has never *seen* the big reward. Pure RL on a laptop GPU is a stretch goal here. The plan is a curriculum from small boards to large ones, so there is a chance to see wins at all, and a non-RL reference (a Hamiltonian-cycle follower) to show what winning looks like.

```widget
summary
```

# 1. Why wins are hard

Three things stack up.

1. **A rare event.** A random policy essentially never fills a board. If a learner never wins, the win bonus never appears in its data, and nothing pulls it toward winning.
2. **A long horizon.** A 6×6 win takes about 300 steps; a 12×12 win takes about 4 800 (the Hamiltonian agent below needs that many). A reward arriving after $T$ steps contributes $\gamma^T R$ to the value of the start state.
3. **Credit assignment.** The mistake that dooms a long snake (coiling into a region too small to survive in) happens many steps before the death, and a cell-level mistake looks identical to a good move at the time.

## The effective horizon

With discount $\gamma$, a reward $T$ steps away is weighted by $\gamma^{T}$, and the weights sum to $1/(1-\gamma)$. That sum is the **effective horizon**: roughly how many steps ahead the agent can "see". With $\gamma = 0.99$ it is 100 steps; with $0.995$, 200; with $0.999$, 1 000.

```widget
horizon_demo
```

A win bonus of 10 after $T = 300$ steps is worth $10\cdot 0.99^{300} \approx 0.5$ at the start with $\gamma = 0.99$ but $10 \cdot 0.995^{300} \approx 2.2$ with $\gamma = 0.995$. This is why the runs here use $\gamma = 0.995$, and why a bigger $\gamma$ is not free: it also raises the variance of every return estimate, which GAE ($\lambda$) then has to control.

# 2. The setup

**Algorithm.** PPO with invalid-action masking, the fastest learner in tryout 04 (masked PPO reached a test mean above 12 within 2 M steps while unmasked PPO was still circling). It uses the batched GPU environment with 256 parallel games.

**Observation: a size-agnostic canvas.** The network sees a fixed 14×14 canvas whatever the board size; the board sits inside it and everything outside is wall. Channels:

| Channel | Meaning | Why |
|---|---|---|
| 0 | tail order: age / length, so how long until each body cell frees | the one thing the snake can *wait for*: a cell next to the tail is safe soon |
| 1 | head | |
| 2 | food | |
| 3 | wall (everything outside the board) | |
| 4 | length / number of cells, a constant plane | tells the net how crowded the board is |
| 5 | *(ablation)* cells reachable from the head through empty cells | the flood-fill feature a planner would compute |

Because the input shape never changes, **the same weights can be trained on 6×6 and then fine-tuned on 8×8, 10×10 and 12×12** (a fully convolutional net would also do this; a fixed canvas is simpler and keeps the tryout-04 architecture).

**Rewards.** +1 per food, −1 on death, and a **win bonus of 10** on filling the board. A starvation timeout stays at $100\times$length steps without food, which already scales with the snake and is far above the cycle time on these boards. A timeout remains a *truncation*, not a terminal, exactly as in tryouts 02 to 04.

# 3. Curriculum learning

Train first on a board where winning is *reachable*, then reuse the weights on a bigger one (Bengio et al. 2009; Narvekar et al. 2020 survey the RL version). Here the "task difficulty" is the board size:

$$6\times6 \;\to\; 8\times8 \;\to\; 10\times10 \;\to\; 12\times12.$$

Why it can help: on 6×6 a decent policy sees wins often enough for the win bonus to *shape* the value function; the 8×8 stage then starts from a policy that already avoids trapping itself. Why it can hurt: weights that fit a small board can encode board-size-specific shortcuts (**negative transfer**), so every stage is compared with a from-scratch control on the same board.

# 4. When learning stalls: a shield with a little planning inside

On 8×8 the plain masked agent plateaued at about 68–73% fill and **never won a single held-out game in 12 M steps**. Its deaths were almost all slow traps: the mask stops the snake from turning into a wall or its body this tick, but not from coiling into a region it cannot leave a dozen moves later. That is a planning failure, and a policy network that sees one board image per step is poorly placed to fix it.

The remedy used here is a **shield**: besides the instant-death mask, the environment also blocks a move when the snake could not survive in the region that move leads to. The check is a *time-aware flood fill* from the cell the head would enter:

* a free cell can be entered at any time; a body cell can be entered at move $m$ once its age is at most $m$ (the tail keeps freeing cells as the snake moves, so most of the body unwinds in time);
* expand the reachable set for $K = \lceil 	ext{cells}/2 
ceil + 2$ moves;
* call the move a **trap** if the reachable area is smaller than the snake's length (one more if the move eats).

If every move is a trap, only instant deaths are blocked; if every move kills, nothing is blocked. The same rule is implemented three times (vectorised on the GPU, in the scalar Python engine, and in the game's JavaScript) and the three agree on every state tested (1 315 states between the GPU version and the Python version; 417 between Python and JavaScript).

**What this means for how to read the result.** A shielded agent is *learning plus a hand-written safety check*. The network still has to learn where to go, but part of what stops it from dying is planning that the environment supplies. It is not pure RL, and the game labels these agents "+ trap shield" so nobody mistakes them for it. The comparison that matters is the same setup with and without the shield, from the same starting weights.

# 5. A planner that never loses: the Hamiltonian cycle

A **Hamiltonian cycle** visits every cell exactly once and returns to its start. A snake that simply follows one can never run into itself (the cell ahead of its head is always either empty or the tail's vacating cell) and passes through every cell, so it eats every food and fills the board, on every seed.

On an even $n\times n$ board a cycle always exists. A simple construction (used in `python/snake_rl/hamilton.py`, and mirrored in the game): run along the top row, zigzag through the remaining rows using columns $1\dots n-1$, then return up the first column.

| Board | Cells | Hamiltonian agent: wins / 500 | mean steps to win |
|---|---|---|---|
| 6×6 | 36 | 500 | about 300 |
| 8×8 | 64 | 500 | about 900 |
| 10×10 | 100 | 500 | about 2 500 |
| 12×12 | 144 | 500 | about 4 800 |

This is the **contrast between planning and learning**. The cycle agent uses the *rules of the world* (it knows the geometry exactly) and no learning at all, and it is perfect and slow (it never takes a shortcut). A learned policy gets no such prior: it has to discover the idea of "do not trap yourself" from a reward that, for a long time, says nothing about it. The agent is in the game's dropdown, clearly labelled *not RL*, as the upper bound to measure everything against. (Real strong Snake players combine the two: they follow a cycle but take *shortcuts* when they are provably safe. Learning such shortcuts on top of the cycle is a natural next experiment, not done here.)

# 6. Results

```widget
curriculum_table
```

```widget
win_curves
```

```widget
win_replay
```

**Analysis.** All numbers are greedy play on the 500 held-out games of that board size, for the checkpoint chosen on 200 separate validation games (the choice prefers more validation wins, then a higher mean). Every run below is a **single training seed**; the win rates fluctuate by 10–20 points from one checkpoint to the next, so read them as ranges.

| Board | Agent (training) | Win rate | Mean fill | Notes |
|---|---|---|---|---|
| 6×6 | masked PPO + reachable-area channel, from scratch, 6 M steps | **97.4%** (487/500) | 98% | 92% already at 2 M steps |
| 6×6 | same without the reachable-area channel (control), 4 M steps | 83.4% (417/500) | 91% | 43% at 2 M steps |
| 8×8 | masked PPO, from scratch, 12 M steps | **88.6%** (443/500) | 91% | first wins at 4 M steps |
| 8×8 | initialised from the 6×6 agent, 12 M steps | 0.8% (4/500) | 42% | the checkpoint chosen on validation is from 2 M steps; later checkpoints fill 70–75% of the board but win 0–1 games |
| 8×8 | 6×6 start + trap shield, 8 M steps | 49.6% (248/500) | 77% | shielded, see §4 |
| 10×10 | masked PPO, from scratch, 16 M steps | 0% at the selected checkpoint (1/500 at 15 M) | 63% | best game 97 of 97 |
| 10×10 | the same run continued for 16 M more steps | **63.0%** (315/500) | 78% | 342/500 at two other checkpoints |
| 12×12 | masked PPO, from scratch, 24 M steps | 0% | 49% | best game 120 of 141 |
| 12×12 | the run above continued (extended run, stopped by the plateau rule at 135 M further steps; validation-selected checkpoint) | **68.0%** (340/500) | 85% | first win at 12.5 M further steps; 70.0% (350/500) at best on the test set |
| any | Hamiltonian cycle (not RL) | 100% | 100% | 6×6 to 12×12 |

1. **Pure reinforcement learning wins small boards reliably, and the win rate drops with the size of the board.** 97% on 6×6, 89% on 8×8, 63% on 10×10 (after 32 M steps in total), and, after the extended run described below, 68% on 12×12 (340 of 500) at the end of writing. The cost of a win grows quickly: a 12×12 win means 141 consecutive correct meals, and the training signal for the last few dozen only arrives in games that have already gone almost perfectly.
2. **The reachable-area channel matters most when training is young.** On 6×6 it roughly doubled the win rate at 2 M steps (92% against 43%) and the gap narrowed to about 10 points by 4 M (94% against 83%). It gives the network, as an input, what a planner would compute (how much free space is still connected to the head). It is information, not a rule, so the agent is still learning; it just learns faster.
3. **Starting from the smaller board hurt on 8×8 in this run.** The agent initialised from the 6×6 weights won 4 of 500 games at its validation-selected checkpoint (2 M steps) and at most one game at any later checkpoint over 12 M steps; the same recipe from scratch won 89%. A fine-tuned policy can carry habits that suit the small board (here, probably a tendency to circle) and be harder to un-learn than to never learn. This is **one seed per variant**, so it is an observation, not a finding about curricula in general; the size-agnostic canvas makes the transfer *possible*, it does not make it *useful*. The fair way to test curriculum learning would be several seeds of each, which this tryout did not run.
4. **The trap shield rescued the curriculum run but did not beat plain learning.** With the shield the 6×6 start reached about 50% on 8×8, far better than without (0.2%), yet below the from-scratch run's 89%. A hand-written safety check removes a failure the learner would otherwise have to discover (§4) at the cost of being part of the agent: the agent is not "pure RL" any more, and the game labels it so. In this experiment it did not turn out to be the best option.
5. **The remaining failures are slow traps.** On 10×10 the final deaths are 114 self-collisions, 30 wall hits and 41 timeouts out of 500; on 12×12, 428 of 500 end by self-collision. The agent usually fills most of the board and then runs out of room: the planner's guarantee, that the route can always be completed, is exactly what it lacks.
6. **More training kept paying on 10×10.** Continuing the 16 M-step run (with the learning rate restarted) took the win rate from 0–1 wins to 47–68% over the next 16 M steps, with a lot of checkpoint-to-checkpoint noise. The first run's learning rate had decayed to zero, which probably capped it.
7. **12×12 was won, after a long extension.** The 24 M-step run did not win a single held-out game (best game 120 of 141 cells). Continuing it with the same recipe, with 1 024 parallel environments instead of 256 (the throughput study in tryout 04, §8), produced the first win after 12.5 M further steps, and the win count then rose quickly:

| Further steps | Mean score | Mean fill | Best game | Wins of 500 |
|---|---|---|---|---|
| 2.5 M | 55.5 | 39% | 112 | 0 |
| 5 M | 71.9 | 51% | 123 | 0 |
| 7.5 M | 65.7 | 47% | 122 | 0 |
| 10 M | 71.4 | 51% | 128 | 0 |
| 12.5 M | 68.7 | 49% | 141 | 1 |
| 15 M | 73.8 | 52% | 141 | 3 |
| 17.5 M | 77.1 | 55% | 136 | 0 |
| 20 M | 85.0 | 60% | 141 | 26 |
| 22.5 M | 84.6 | 60% | 141 | 40 |
| 25 M | 91.9 | 65% | 141 | 112 |
| 37.5 M | 98.4 | 70% | 141 | 195 |
| 50 M | 109.2 | 77% | 141 | 258 |
| 62.5 M | 114.2 | 81% | 141 | 281 |
| 80 M | 116.3 | 83% | 141 | 312 |
| 90 M | 118.7 | 84% | 141 | 333 |
| 107.5 M | 120.1 | 85% | 141 | 350 (highest) |
| 125 M | 120.1 | 85% | 141 | 340 (published) |
| 135 M (end) | 112.7 | 80% | 141 | 303 |

This is a single run (one seed). The win rate rose quickly up to about 50 M further steps, then slowly, and from about 90 M further steps it moved between 55% and 70% from one evaluation to the next (the standard error of a win rate measured on 500 games is about 2 percentage points, and the checkpoints themselves differ, so swings of this size are partly noise and partly real changes in the policy). The plateau rule stopped the run at 135 M further steps because neither the fill (85.1% against 85.2% before) nor the win rate (68.2% against 70.0%) had gained over the last 25 M steps. The published agent is the checkpoint with the best validation score (340 of 500 test wins, 68.0%); the single best test evaluation of the run was 350 of 500 (70.0%), which is not used as the headline because choosing the maximum over 50 noisy evaluations would flatter the result. Almost all remaining failures are self-collisions after the agent has filled most of the board (140 of 500 at the published checkpoint). The full evaluation-by-evaluation table is in `docs/12x12_trajectory.md`. What it shows is that pure reinforcement learning, a masked PPO policy with a reachable-area input, *can* fill the full 12×12 board on a laptop GPU, but it took on the order of 150 M environment steps in total for about two-thirds of the games, against none at all for the Hamiltonian cycle's route to the same result (which learns nothing).

**What the Hamiltonian cycle adds to this picture.** It reaches 100% on every size with zero learning, but it takes the long way (about 4 800 steps for a 12×12 win), and it needs the rules and an even board. The learned agents are shorter and more general in what they need to be told, but they are not close to reliable beyond 8×8. Neither is "better": they answer different questions (what can be *learned* from reward, and what can be *computed* with the rules).

# 7. What to try next

* **Search on top of the learned value function** (AlphaZero-style tree search, or even a short rollout with the flood-fill check): planning supplies the lookahead that the network lacks.
* **Imitation, then RL**: pretrain on the Hamiltonian agent's trajectories (a strong, safe policy) and let RL improve on its speed.
* **Reward shaping by free area**: a potential $\Phi(s) = $ reachable-area fraction, which keeps the optimal policy (Ng et al. 1999) while rewarding the behaviour a planner would.
* **More compute**: the hardest stages are limited by training time on a 1650 Ti, not by the idea.

# 8. Failure modes to watch for

* **Win rate hides a collapse in fill.** Track mean length and fill % too, and the causes of death.
* **Catastrophic forgetting between stages**: re-evaluate the new weights on the *previous* board size.
* **A mask is not planning.** It stops instant deaths only; the main killer of long snakes is the slow trap.
* **Counting a lucky seed as a result.** The win rate is over 500 held-out seeds, and each stage has a from-scratch control.

# Self-check

```quiz
Q: With γ = 0.99 and a win bonus of 10 arriving 400 steps after a decision, roughly how much does that bonus add to the value of the decision?
- [ ] About 10
- [x] About 0.18, since 0.99^400 ≈ 0.018
- [ ] Exactly zero
Why: Rewards are weighted by γ^T. At γ = 0.99 the effective horizon is only 100 steps, so a distant bonus is almost invisible, which is why a larger γ is used for this task.
```

```quiz
Q: Why can a snake that follows a Hamiltonian cycle never trap itself?
- [ ] Because it moves very fast
- [x] The cycle visits every cell once; the snake's body always occupies a contiguous stretch of it, so the next cell is empty or the vacating tail
- [ ] Because it avoids food
Why: Knowing the whole geometry in advance is planning. The agent needs no learning, only the rules.
```

# References

- Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *ICML*.
- Narvekar, S., Peng, B., Leonetti, M., Sinapov, J., Taylor, M. E., & Stone, P. (2020). Curriculum learning for reinforcement learning domains: a framework and survey. *JMLR*, 21.
- Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: theory and application to reward shaping. *ICML*.
- Pardo, F., Tavakoli, A., Levdik, V., & Kormushev, P. (2018). Time limits in reinforcement learning. *ICML*.
- Schulman, J., et al. (2016). High-dimensional continuous control using generalized advantage estimation. *ICLR*.
- Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
