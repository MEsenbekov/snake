import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = 1.0  # Начальное значение для ε-greedy
        self.epsilon_min = 0.01  # Минимальное значение ε
        self.epsilon_decay = 0.995  # Снижение ε после каждого шага
        self.memory = deque(maxlen=2000)  # Память для опыта
        self.batch_size = 32
        self.gamma = 0.99  # Коэффициент дисконтирования

    def build_model(self):
        # Измените размерность входа на state_size (например, 7)
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),  # Входная размерность должна быть state_size (например, 7)
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def act(self, state):
        """Выбор действия с использованием стратегии ε-greedy"""
        if np.random.rand() <= self.epsilon:
            # Случайный выбор действия (исследование)
            return random.randrange(self.action_size)
        # Выбор действия с максимальным значением Q (эксплуатация)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """Запоминаем опыт"""
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, state, action, reward, next_state, done):
        """Обучение модели на основе опыта"""
        self.remember(state, action, reward, next_state, done)

        if len(self.memory) < self.batch_size:
            return

        # Получаем случайную выборку из памяти
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.BoolTensor(dones)

        # Получаем Q-значения для текущего состояния
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # Ожидаемое Q-значение
        next_q_values = self.model(next_states).max(1)[0].detach()
        expected_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Вычисляем потерю и обновляем параметры модели
        loss = self.criterion(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Снижаем ε, чтобы уменьшить случайность действий
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_progress(self, filename="dqn_agent.pth"):
        """Сохранение прогресса обучения"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': self.memory
        }, filename)
        print(f"Progress saved to {filename}")

    def load_progress(self, filename="dqn_agent.pth"):
        """Загрузка прогресса обучения"""
        try:
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.memory = checkpoint['memory']
            print(f"Progress loaded from {filename}")
        except FileNotFoundError:
            print("No saved progress found, starting from scratch.")