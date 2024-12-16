import pygame
import random
import numpy as np
import torch
from dqn_agent import DQNAgent
from snake_game import SnakeGame  # Подключаем игру


def main():
    state_size = 6  # Например, количество признаков в состоянии
    action_size = 4  # Количество возможных действий (вверх, вниз, влево, вправо)

    agent = DQNAgent(state_size, action_size)

    # Попытаться загрузить прогресс обучения, если он есть
    agent.load_progress()

    for episode in range(1000):  # Например, 1000 эпизодов обучения
        game = SnakeGame()  # Создаем объект игры
        state = game.get_state()

        while not game.done:
            action = agent.act(state)  # Агент выбирает действие
            next_state, reward, done = game.step(action)  # Выполняем шаг игры
            agent.learn(state, action, reward, next_state, done)  # Агент обучается
            state = next_state

        # Сохраняем прогресс после каждого эпизода или через определенные интервалы
        if episode % 10 == 0:
            agent.save_progress()  # Сохраняем прогресс каждое 10-е обновление

    # Сохраняем прогресс после завершения всех эпизодов
    agent.save_progress()


if __name__ == "__main__":
    main()