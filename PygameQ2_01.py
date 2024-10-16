import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Side-Scrolling Game")
clock = pygame.time.Clock()

# Fonts
font = pygame.font.SysFont('comicsans', 30)

# Player class
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((50, 50))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.x = 100
        self.rect.y = SCREEN_HEIGHT - 150
        self.speed = 5
        self.jump_speed = 12
        self.is_jumping = False
        self.velocity_y = 0
        self.health = 100
        self.lives = 3
        self.score = 0

    def move(self, dx, dy):
        self.rect.x += dx
        self.rect.y += dy

    def jump(self):
        if not self.is_jumping:
            self.is_jumping = True
            self.velocity_y = -self.jump_speed

    def update(self):
        # Gravity
        if self.is_jumping:
            self.velocity_y += 1
            self.rect.y += self.velocity_y
            if self.rect.y >= SCREEN_HEIGHT - 150:
                self.rect.y = SCREEN_HEIGHT - 150
                self.is_jumping = False

        if self.health <= 0:
            self.lives -= 1
            self.health = 100

    def shoot(self):
        bullet = Projectile(self.rect.centerx, self.rect.y)
        bullets.add(bullet)

    def draw_health_bar(self):
        pygame.draw.rect(screen, RED, (self.rect.x, self.rect.y - 10, 50, 10))
        pygame.draw.rect(screen, GREEN, (self.rect.x, self.rect.y - 10, 50 * (self.health / 100), 10))

# Projectile class
class Projectile(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((10, 5))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = 10
        self.damage = 10

    def update(self):
        self.rect.x += self.speed
        if self.rect.x > SCREEN_WIDTH:
            self.kill()

# Collectible class
class Collectible(pygame.sprite.Sprite):
    def __init__(self, x, y, type):
        super().__init__()
        self.image = pygame.Surface((30, 30))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.type = type

    def apply_effect(self, player):
        if self.type == "health":
            player.health += 20
            if player.health > 100:  # Prevent exceeding max health
                player.health = 100
        elif self.type == "life":
            player.lives += 1
        self.kill()

# Enemy class with health bar and defeat points
class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((50, 50))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = 2
        self.health = 50
        self.movement_type = random.choice(["patrol", "follow"])  # Patrol or follow the player
        self.direction = 1  # 1 for right, -1 for left

    def move(self, player):
        if self.movement_type == "patrol":
            self.rect.x += self.speed * self.direction
            if self.rect.x >= SCREEN_WIDTH - 50 or self.rect.x <= 0:
                self.direction *= -1
        elif self.movement_type == "follow":
            if abs(self.rect.x - player.rect.x) < 200:
                if player.rect.x > self.rect.x:
                    self.rect.x += self.speed
                elif player.rect.x < self.rect.x:
                    self.rect.x -= self.speed

    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            player.score += 100  # Add points for defeating the enemy
            self.kill()

    def update(self, player):
        self.move(player)

    def draw_health_bar(self):
        pygame.draw.rect(screen, RED, (self.rect.x, self.rect.y - 10, 50, 10))
        pygame.draw.rect(screen, GREEN, (self.rect.x, self.rect.y - 10, 50 * (self.health / 50), 10))

# BossEnemy class
class BossEnemy(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((100, 100))
        self.image.fill((255, 215, 0))  # Gold color
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.health = 300
        self.speed = 3

    def move(self, player):
        if self.rect.x < player.rect.x:
            self.rect.x += self.speed
        else:
            self.rect.x -= self.speed

    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            player.score += 500  # Add points for defeating the boss
            self.kill()

    def update(self, player):
        self.move(player)

    def draw_health_bar(self):
        pygame.draw.rect(screen, RED, (self.rect.x, self.rect.y - 10, 100, 10))
        pygame.draw.rect(screen, GREEN, (self.rect.x, self.rect.y - 10, 100 * (self.health / 300), 10))

# Function to draw the player's score on the screen
def draw_score(player):
    score_text = font.render(f"Score: {player.score}", True, WHITE)
    screen.blit(score_text, (10, 10))

# Function to set up levels
def setup_level(level):
    enemies.empty()
    collectibles.empty()
    all_sprites.empty()

    if level == 1:
        for i in range(3):  # 3 enemies in level 1
            enemy = Enemy(random.randint(400, 800), SCREEN_HEIGHT - 150)
            enemies.add(enemy)
            all_sprites.add(enemy)

        collectible = Collectible(500, SCREEN_HEIGHT - 100, 'health')
        collectibles.add(collectible)
        all_sprites.add(collectible)

    elif level == 2:
        for i in range(5):  # 5 enemies in level 2
            enemy = Enemy(random.randint(400, 800), SCREEN_HEIGHT - 150)
            enemies.add(enemy)
            all_sprites.add(enemy)

        collectible = Collectible(300, SCREEN_HEIGHT - 100, 'life')
        collectibles.add(collectible)
        all_sprites.add(collectible)

    elif level == 3:
        # Level 3: Boss Fight
        global boss  # Declare boss as global
        boss = BossEnemy(600, SCREEN_HEIGHT - 150)
        all_sprites.add(boss)
        for i in range(2):  # 2 additional enemies
            enemy = Enemy(random.randint(400, 800), SCREEN_HEIGHT - 150)
            enemies.add(enemy)
            all_sprites.add(enemy)

    all_sprites.add(player)

# Game loop
def main():
    run = True
    global player
    player = Player()
    global bullets
    bullets = pygame.sprite.Group()
    global enemies
    enemies = pygame.sprite.Group()
    global collectibles
    collectibles = pygame.sprite.Group()
    global all_sprites
    all_sprites = pygame.sprite.Group()  # Define all_sprites here

    current_level = 1
    setup_level(current_level)

    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        keys = pygame.key.get_pressed()
        # Player controls
        if keys[pygame.K_LEFT]:
            player.move(-player.speed, 0)
        if keys[pygame.K_RIGHT]:
            player.move(player.speed, 0)
        if keys[pygame.K_SPACE]:
            player.shoot()
        if keys[pygame.K_UP]:
            player.jump()

        # Update player
        player.update()

        # Update bullets
        bullets.update()

        # Update all enemies and boss
        for enemy in enemies:
            enemy.update(player)
        if current_level == 3 and 'boss' in locals():
            boss.update(player)

        # Handle collisions
        for bullet in bullets:
            for enemy in enemies:
                if pygame.sprite.collide_rect(bullet, enemy):
                    enemy.take_damage(bullet.damage)
                    bullet.kill()  # Remove bullet on hit

        # Check for player collision with enemies
        for enemy in enemies:
            if pygame.sprite.collide_rect(enemy, player):
                player.health -= 10
                enemy.kill()  # Remove enemy after it hits the player

        # Check for player death
        if player.lives <= 0:
            run = False  # End the game loop if no lives left

        # Check for boss defeat
        if current_level == 3 and boss.health <= 0:
            current_level += 1
            if current_level > 3:  # If past last level, game won
                run = False
            else:
                setup_level(current_level)  # Load next level

        # Check for collectible collision
        for collectible in collectibles:
            if pygame.sprite.collide_rect(collectible, player):
                collectible.apply_effect(player)

        # Draw everything
        screen.fill(BLACK)
        all_sprites.draw(screen)
        bullets.draw(screen)
        collectibles.draw(screen)

        # Draw score and health bars
        draw_score(player)
        player.draw_health_bar()

        for enemy in enemies:
            enemy.draw_health_bar()

        if current_level == 3 and 'boss' in locals():
            boss.draw_health_bar()

        # Update the display
        pygame.display.flip()

    # Game Over screen
    screen.fill(BLACK)
    game_over_text = font.render("GAME OVER", True, WHITE)
    restart_text = font.render("Press R to Restart or Q to Quit", True, WHITE)
    screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - 20))
    screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2 + 10))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    main()  # Restart the game
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()

if __name__ == "__main__":
    main()
