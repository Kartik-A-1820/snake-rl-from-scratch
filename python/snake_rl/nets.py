# Q-networks. Kept as plain nn.Sequential blocks so export_model.py can walk them layer by layer
# into the JSON format the browser game runs (game/index.html: runLayers).
import torch
from torch import nn


class QNet(nn.Module):
    def __init__(self, body, feat_dim, n_actions=3, hidden=256, dueling=False):
        super().__init__()
        self.body, self.dueling = body, dueling
        if dueling:
            # Wang et al. 2016: Q = V + (A - mean A); identifiable and learns V from every action's update
            self.value = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.adv = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
        else:
            self.head = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x):
        h = self.body(x)
        if self.dueling:
            a = self.adv(h)
            return self.value(h) + a - a.mean(1, keepdim=True)
        return self.head(h)


def grid_net(size=12, dueling=False, channels=(32, 64, 64), hidden=256, in_ch=4):
    # Input (4, size+2, size+2). Stride-2 convs shrink 14x14 → 7x7 → 4x4 so the dense layer stays small.
    c1, c2, c3 = channels
    body = nn.Sequential(
        nn.Conv2d(in_ch, c1, 3, 1, 1), nn.ReLU(),
        nn.Conv2d(c1, c2, 3, 2, 1), nn.ReLU(),
        nn.Conv2d(c2, c3, 3, 2, 1), nn.ReLU(),
        nn.Flatten(),
    )
    with torch.no_grad():
        feat = body(torch.zeros(1, in_ch, size + 2, size + 2)).shape[1]
    return QNet(body, feat, hidden=hidden, dueling=dueling)


def mlp_net(in_dim=11, hidden=64, dueling=False):
    body = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Flatten())
    return QNet(body, hidden, hidden=hidden, dueling=dueling)


def make_net(obs="grid", size=12, dueling=False):
    return grid_net(size, dueling) if obs == "grid" else mlp_net(11, 64, dueling)


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def size_bytes(net, bytes_per_param=4):
    return n_params(net) * bytes_per_param


class ActorCritic(nn.Module):
    # PPO network (Schulman et al. 2017): one conv trunk shared by the policy and the value function, then separate
    # heads. forward(x) returns the policy logits only, so the export / ONNX / game code can treat it like a Q-net
    # (the game takes the argmax of the outputs); forward_ac(x) returns (logits, value) for training.
    # Orthogonal init (gain sqrt 2 in the trunk, 0.01 on the policy output so the first policy is almost uniform,
    # 1.0 on the value output) follows Engstrom et al. 2020 / Andrychowicz et al. 2021.
    dueling = False

    def __init__(self, size=12, hidden=256, in_ch=4):
        super().__init__()
        base = grid_net(size, in_ch=in_ch)
        self.body = base.body
        feat = base.head[0].in_features
        self.head = nn.Sequential(nn.Linear(feat, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.value_head = nn.Sequential(nn.Linear(feat, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, 2 ** 0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.head[-1].weight, 0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, 1.0)

    def forward(self, x):
        return self.head(self.body(x))

    def forward_ac(self, x):
        h = self.body(x)
        return self.head(h), self.value_head(h).squeeze(-1)

    def value(self, x):
        return self.value_head(self.body(x)).squeeze(-1)
