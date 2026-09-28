# RL weights / RL 权重

Two PPO residual policies trained **in MiniSim** (our own simulator, `py/minisim/rl.py`), both tested in the real game.
Both **lost** there. They are published so the result can be checked and reproduced, not as a better bot.
两个在 **MiniSim**（自建模拟器，`py/minisim/rl.py`）里用 PPO 训练的残差策略，都拿进真游戏测过，**都亏了**。
公开它们是为了让结果可以核对和复现，不是因为它们打得更好。

| File | Actions (every 0.5 s, residual on the ghost tracker) | MiniSim vs zero residual | Real game, 32 v 32 |
|---|---|---|---|
| `minisim-ppo-s1-it050.pt` | lateral offset + time warp (`act_dims` 2) | +37.7 ± 3.2 | **−32.0 ± 6.1** (21 of 32 matches jammed balls against the red bump face) |
| `minisim-ppo-s2-it105.pt` | lateral offset only, clearance constraint (`act_dims` 1) | +51.6 ± 2.9 | **−32.9 ± 7.0** (no jams; mean offset only 5 cm) |

- **Network** (`py/mosimrl/nets.py` `ResidualNet`): two CNNs over ball-count grids (0.5 m field grid and 0.25 m robot-frame grid; floor and airborne channels, hub interior masked) plus an MLP over the ghost's upcoming path, robot / match / phase features and a 17-lane swath yield (`py/mosimrl/obs.py` `residual_obs`). The actor's mean layer starts at zero, so training starts from plain tracking.
- **Training**: PPO, 64 episodes per iteration on 16 processes, domain randomization over 10 simulator parameters, potential-based shaping, γ 0.99 per decision, λ 0.95. The hyper-parameters are stored in each checkpoint (`args`).
- **Format**: `torch.load(...)` gives `{"net": state_dict, "it": iteration, "k": 2, "args": {...}}`. Checksums are in `SHA256SUMS`.
- **Why they failed**: MiniSim tracks the plain route within about 20 points, but it gets the sign of small residuals around the route wrong. See `docs/research-log/proposals.md` (entries c50 and d105).
- **网络**：见 `py/mosimrl/nets.py` 的 `ResidualNet`。
  - 两路 CNN 看球的计数网格：0.5 m 全场网格和 0.25 m 以车为中心的网格，分地面和空中两个通道，hub 内部屏蔽。
  - 一个 MLP 看鬼影路线接下来的轨迹、车况 / 比赛 / 阶段特征，以及 17 条扫带的可吸球数。
  - 均值层零初始化，训练从普通跟车出发。
- **失败原因**：MiniSim 对原路线的总分只差约 20 分，但对“路线附近的小改动”连方向都算反。详见 `docs/research-log/proposals.md`（c50、d105 两条）。

Run one against the plain tracker in the real game (A/B, arms alternate):
在真游戏里和普通跟车做 A/B（两种策略交替）：

```bash
cd py && python -m mosimrl.run_ghost --demo ../run/demos/demo-20260926-015440-m1.jsonl --ports 47500,47501 \
  --episodes 8 --time-scale 1 --steps-per-frame 2 --arms \
  '{"chooser":"net","ckpt":"../weights/minisim-ppo-s2-it105.pt","leash":{"neutral":0.333,"conveyor":0.333,"harvest":0.5,"shoot":0.5},"decide_every":5,"act_dims":1,"clearance":true};{"chooser":"zero","leash":{"neutral":0.333,"conveyor":0.333,"harvest":0.5,"shoot":0.5},"decide_every":5}'
```

For `minisim-ppo-s1-it050.pt`, drop `"act_dims":1,"clearance":true`.
用 `minisim-ppo-s1-it050.pt` 时，去掉 `"act_dims":1,"clearance":true`。

The A/Bs above used the default game threads (before `-job-worker-count 0` was adopted).
上表的 A/B 用的是游戏默认线程数（当时还没改成 `-job-worker-count 0`）。
