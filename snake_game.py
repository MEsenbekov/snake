import pygame
import random
import numpy as np

# Размеры экрана и блоков
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
BLOCK_SIZE = 20

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Временные ограничения
MAX_TIME_WITHOUT_EATING = 30000  # 30 секунд
MAX_TIME_WITHOUT_EATING_FROM_START = 60000  # 1 минута с начала игры

class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        pygame.font.init()  # Инициализация шрифтов
        self.reset()

    def reset(self):
        self.snake = [(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)]
        self.snake_dir = (BLOCK_SIZE, 0)
        self.food = self.random_food()
        self.score = 0
        self.done = False
        self.snake_length = 1
        self.steps = 0
        self.time_food_appeared = pygame.time.get_ticks()  # Время появления последней ягоды
        self.start_time = pygame.time.get_ticks()  # Время начала игры
        self.food_eaten = 0  # Счетчик съеденных ягод

    def random_food(self):
        return (random.randrange(0, SCREEN_WIDTH, BLOCK_SIZE),
                random.randrange(0, SCREEN_HEIGHT, BLOCK_SIZE))

    def move_snake(self):
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.snake_dir
        new_head = (head_x + dir_x, head_y + dir_y)

        self.snake = [new_head] + self.snake[:-1]

        if new_head == self.food:
            # Если змея съела ягоду, она удлиняется
            self.snake.append(self.snake[-1])
            self.food = self.random_food()
            self.snake_length += 1
            self.score += 1
            self.food_eaten += 1
            self.time_food_appeared = pygame.time.get_ticks()  # Обновляем время появления новой ягоды

    def is_collision(self):
        head_x, head_y = self.snake[0]
        if head_x < 0 or head_x >= SCREEN_WIDTH or head_y < 0 or head_y >= SCREEN_HEIGHT:
            return True
        if (head_x, head_y) in self.snake[1:]:
            return True
        return False

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        dir_x, dir_y = self.snake_dir

        # Возвращаем состояние: позиция головы змеи, позиция еды и направление
        return np.array([head_x, head_y, food_x, food_y, dir_x, dir_y, self.snake_length], dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.snake_dir = (0, -BLOCK_SIZE)  # вверх
        elif action == 1:
            self.snake_dir = (0, BLOCK_SIZE)  # вниз
        elif action == 2:
            self.snake_dir = (-BLOCK_SIZE, 0)  # влево
        elif action == 3:
            self.snake_dir = (BLOCK_SIZE, 0)  # вправо

        self.move_snake()

        # Проверка на смерть
        self.check_death_conditions()

        self.steps += 1
        reward = 0.1  # Награда за каждый шаг

        # Срок жизни змеи: чем дольше она живет, тем больше награда
        lifetime_reward = self.steps / 100  # Можно настроить множитель

        # Количество съеденных ягод
        food_reward = self.food_eaten * 2  # Можно настроить множитель

        # Скорость поедания ягод
        time_taken = pygame.time.get_ticks() - self.time_food_appeared
        speed_reward = max(0, 1 - (time_taken / 1000))  # За скорость поедания (чем меньше время, тем лучше)

        # Общая награда
        reward += lifetime_reward + food_reward + speed_reward

        if self.done:
            reward -= 1  # Штраф за смерть

        return self.get_state(), reward, self.done

    def check_death_conditions(self):
        # Проверка на смерть из-за времени без еды
        time_without_eating = pygame.time.get_ticks() - self.time_food_appeared
        if time_without_eating > MAX_TIME_WITHOUT_EATING:
            self.done = True

        # Проверка на смерть если змея не съела ягоду в течение 1 минуты с начала игры
        time_from_start = pygame.time.get_ticks() - self.start_time
        if time_from_start > MAX_TIME_WITHOUT_EATING_FROM_START and self.food_eaten == 0:
            self.done = True

    def render(self):
        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, RED, (self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, (segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE))

        font = pygame.font.SysFont("Arial", 24)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()

    def run(self):
        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True

            self.step(random.randint(0, 3))  # Для теста случайное действие
            self.render()
            self.clock.tick(10)

        return self.score


def main():
    game = SnakeGame()  # Создаем объект игры

    # Убедитесь, что вызываете get_state после того, как объект был полностью инициализирован
    state = game.get_state()
    print("Initial state:", state)

    game.run()


if __name__ == "__main__":
    main()