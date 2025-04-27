import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
BALL_RADIUS = 10
PLAYER_SPEED = 5

# Initialize game objects
player1 = pygame.Rect(50, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
player2 = pygame.Rect(WIDTH - 50 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
ball = pygame.Rect(WIDTH // 2 - BALL_RADIUS, HEIGHT // 2 - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)
ball_speed_x = 5 * random.choice([-1, 1])  # Random initial direction
ball_speed_y = 5 * random.choice([-1, 1])


screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")
clock = pygame.time.Clock()


def player_movement(player, direction):
    if direction == "up" and player.top > 0:
        player.y -= PLAYER_SPEED
    if direction == "down" and player.bottom < HEIGHT:
        player.y += PLAYER_SPEED


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                player_movement(player1, "up")
            if event.key == pygame.K_s:
                player_movement(player1, "down")
            if event.key == pygame.K_UP:
                player_movement(player2, "up")
            if event.key == pygame.K_DOWN:
                player_movement(player2, "down")
    

    # Move the ball
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # Ball collisions with walls
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_speed_y *= -1
    if ball.left <= 0 or ball.right >= WIDTH:
      ball_speed_x *= -1
    
    # Ball collisions with paddles
    if ball.colliderect(player1) or ball.colliderect(player2):
        ball_speed_x *= -1


    # Update display
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, player1)
    pygame.draw.rect(screen, WHITE, player2)
    pygame.draw.circle(screen, WHITE, (ball.x + BALL_RADIUS, ball.y + BALL_RADIUS), BALL_RADIUS)
    pygame.display.flip()
    clock.tick(FPS)


pygame.quit()