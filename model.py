import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import sounddevice as sd

# ===============================
# Simplified DDPG model (reduced)
# ===============================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ===============================
# Simplified DDPG Agent
# ===============================
class DDPGAgent:
    def __init__(self, state_dim, action_dim, gamma=0.98, tau=0.005):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        with torch.no_grad():
            return self.actor(torch.FloatTensor(state)).numpy()

    def train_step(self, replay_buffer, batch_size=32):
        if len(replay_buffer) < batch_size:
            return

        batch = np.random.choice(len(replay_buffer), batch_size, replace=False)
        state, action, reward, next_state = zip(*[replay_buffer[i] for i in batch])

        state = torch.FloatTensor(np.array(state))
        action = torch.FloatTensor(np.array(action))
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(next_state))

        with torch.no_grad():
            target_action = self.actor_target(next_state)
            target_q = reward + self.gamma * self.critic_target(next_state, target_action)

        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ===============================
# Main Program
# ===============================
def main():
    # Load your dataset
    df = pd.read_csv('merged_instruments_dataset(2028 audio files).csv')

    # Drop non-numeric columns automatically (like file paths, text)
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        raise ValueError("No numeric columns found — please check dataset features.")

    # Keep emotion and instrument separately
    if 'emotion' not in df.columns or 'instrument' not in df.columns:
        raise ValueError("Dataset must contain 'emotion' and 'instrument' columns!")

    # Work with smaller sample for easier testing
    df = df.sample(n=min(100, len(df)), random_state=42).reset_index(drop=True)
    numeric_df = df.select_dtypes(include=[np.number])

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(numeric_df)
    df[numeric_df.columns] = scaled_features

    emotions = ['happy', 'sad', 'angry', 'disgust', 'fear', 'surprise', 'shocked']
    instruments = ['sitar', 'veena', 'tabla', 'flute', 'mohanveena', 'harmonium']

    print('Available emotions:', emotions)
    emotion = input('Enter emotion: ').strip().lower()
    print('Available instruments:', instruments)
    instrument = input('Enter instrument: ').strip().lower()

    subset = df[(df['emotion'] == emotion) & (df['instrument'] == instrument)]
    if len(subset) == 0:
        raise ValueError(f"No matching data found for emotion '{emotion}' and instrument '{instrument}'")

    if len(subset) > 20:
        subset = subset.sample(20, random_state=42)

    features = numeric_df.columns.tolist()
    state_dim = len(features)
    action_dim = len(features)

    agent = DDPGAgent(state_dim, action_dim)
    replay_buffer = []

    print(f'Training DDPG model for emotion={emotion}, instrument={instrument}...')

    for step in range(500):  # reduced training steps
        state = subset.sample(1)[features].values[0]
        action = agent.select_action(state)
        reward = -np.mean((action - state) ** 2)
        next_state = subset.sample(1)[features].values[0]
        replay_buffer.append((state, action, reward, next_state))
        agent.train_step(replay_buffer)

        if step % 100 == 0:
            print(f'Training step {step}/500...')

    print('Training complete. Generating audio...')
    final_action = agent.select_action(state)

    # Generate simple audio wave based on learned pattern
    duration = 2  # seconds
    samplerate = 44100
    freq = 440 + final_action[0] * 100
    t = np.linspace(0, duration, int(samplerate * duration))
    audio = np.sin(2 * np.pi * freq * t)

    sd.play(audio, samplerate=samplerate)
    sd.wait()
    print('✅ Audio generation complete!')

if __name__ == '__main__':
    main()

