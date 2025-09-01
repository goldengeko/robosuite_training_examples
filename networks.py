import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------
# Shared Actor & Critic for TD3
# -------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        q1 = F.relu(self.l1(xu))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(xu))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        q1 = F.relu(self.l1(xu))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)

# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) < self.max_size:
            self.storage.append(data)
        else:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in ind:
            s, a, r, s2, d = self.storage[i]
            states.append(np.asarray(s))
            actions.append(np.asarray(a))
            rewards.append(np.asarray(r))
            next_states.append(np.asarray(s2))
            dones.append(np.asarray(d))

        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)
        )


# -------------------------
# TD3 (Off-Policy)
# -------------------------
class TD3:
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.discount * target_Q)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# -------------------------
# PPO (On-Policy)
# -------------------------
class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.base1 = nn.Linear(state_dim, 400)
        self.base2 = nn.Linear(400, 300)
        self.mu = nn.Linear(300, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.base1(state))
        x = F.relu(self.base2(x))
        mu = torch.tanh(self.mu(x)) * self.max_action
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def act(self, state):
        mu, std = self.forward(state)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
        return action.clamp(-self.max_action, self.max_action), log_prob

class ValueCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

class PPO:
    def __init__(self, state_dim, action_dim, max_action,
                 clip_ratio=0.2, lr=3e-4, gamma=0.99, lam=0.95,
                 train_iters=80, target_kl=0.01):
        self.actor = PPOActor(state_dim, action_dim, max_action).to(device)
        self.critic = ValueCritic(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam
        self.train_iters = train_iters
        self.target_kl = target_kl

    def compute_advantages(self, rewards, values, dones):
        adv = torch.zeros_like(rewards)
        lastgaelam = 0
        values = torch.cat([values, torch.zeros(1)])  # bootstrap last value
        for t in reversed(range(len(rewards))):
            nextnonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * nextnonterminal - values[t]
            adv[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        return adv, adv + values[:-1]

    def update(self, states, actions, log_probs_old, returns, advantages):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.train_iters):
            mu, std = self.actor(states)
            dist = Normal(mu, std)
            log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)
            ratio = (log_probs - log_probs_old).exp()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            value = self.critic(states)
            critic_loss = F.mse_loss(value, returns)

            entropy = dist.entropy().mean()

            self.actor_optimizer.zero_grad()
            (actor_loss - 0.01 * entropy).backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            approx_kl = (log_probs_old - log_probs).mean().item()
            if approx_kl > 1.5 * self.target_kl:
                break

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            action, log_prob = self.actor.act(state)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()
