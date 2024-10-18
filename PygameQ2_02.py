import pygame
import sys
import random

# Initialize Pygame and the mixer for sound
pygame.init()
pygame.mixer.init()

# Load the background music
pygame.mixer.music.load('/Users/alesa/Desktop/Assignments3_Q2/assets/background_music.mp3') 
pygame.mixer.music.play(-1)  # Play the music indefinitely

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

# Load the background image
background_image = pygame.image.load("/Users/alesa/Desktop/Assignments3_Q2/background.png").convert()
background_width = background_image.get_width()

# Camera class for dynamic camera
class Camera:
    def __init__(self, width, height):
        self.camera = pygame.Rect(0, 0, width, height)
        self.width = width
        self.height = height

    def apply(self, entity):
        return entity.rect.move(self.camera.topleft)

    def update(self, target):
        x = -target.rect.centerx + int(SCREEN_WIDTH / 2)
        y = -target.rect.centery + int(SCREEN_HEIGHT / 2)
        x = min(0, x)  # Left boundary
        x = max(-(self.width - SCREEN_WIDTH), x)  # Right boundary
        y = min(0, y)  # Top boundary
        y = max(-(self.height - SCREEN_HEIGHT), y)  # Bottom boundary
        self.camera = pygame.Rect(x, y, self.width, self.height)

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
        self.jump_speed = 15
        self.gravity = 0.8
        self.velocity_y = 0
        self.is_jumping = False
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
        self.velocity_y += self.gravity
        self.rect.y += self.velocity_y
        if self.rect.y >= SCREEN_HEIGHT - 150:
            self.rect.y = SCREEN_HEIGHT - 150
            self.velocity_y = 0
            self.is_jumping = False

        if self.health <= 0:
            self.lives -= 1
            self.health = 100

    def shoot(self):
        bullet = Projectile(self.rect.centerx, self.rect.y, RED, 10)
        bullets.add(bullet)

    def draw_health_bar(self):
        pygame.draw.rect(screen, RED, (self.rect.x, self.rect.y - 10, 50, 10))
        pygame.draw.rect(screen, GREEN, (self.rect.x, self.rect.y - 10, 50 * (self.health / 100), 10))

# Projectile class
class Projectile(pygame.sprite.Sprite):
    def __init__(self, x, y, color, damage):
        super().__init__()
        self.image = pygame.Surface((10, 5))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = 10
        self.damage = damage

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
            if player.health > 100:
                player.health = 100
        elif self.type == "life":
            player.lives += 1
        self.kill()

# Enemy class
class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((50, 50))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = 2
        self.health = 150  # Increased health to make enemies harder to defeat
        self.movement_type = random.choice(["patrol", "follow"])
        self.direction = 1

    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            player.score += 100
            self.kill()

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
            player.score += 100
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
        self.image.fill((255, 215, 0))  # Gold color for boss
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
            player.score += 500
            self.kill()

    def update(self, player):
        self.move(player)

    def draw_health_bar(self):
        pygame.draw.rect(screen, RED, (self.rect.x, self.rect.y - 10, 100, 10))
        pygame.draw.rect(screen, GREEN, (self.rect.x, self.rect.y - 10, 100 * (self.health / 300), 10))

# Function to display a transition message
def level_transition_message(screen, message, font):
    screen.fill(BLACK)
    text = font.render(message, True, WHITE)
    screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 - text.get_height() // 2))
    pygame.display.flip()
    pygame.time.delay(2000)

# Setup level function with boss creation
def setup_level(level):
    # Show level start message
    level_transition_message(screen, f"Level {level} Start", font)

    enemies.empty()
    collectibles.empty()
    all_sprites.empty()

    if level == 1:
        for i in range(3):
            enemy = Enemy(random.randint(400, 800), SCREEN_HEIGHT - 150)
            enemies.add(enemy)
            all_sprites.add(enemy)
        collectible = Collectible(500, SCREEN_HEIGHT - 100, 'health')
        collectibles.add(collectible)
        all_sprites.add(collectible)

    elif level == 2:
        for i in range(5):
            enemy = Enemy(random.randint(400, 800), SCREEN_HEIGHT - 150)
            enemies.add(enemy)
            all_sprites.add(enemy)
        collectible = Collectible(300, SCREEN_HEIGHT - 100, 'life')
        collectibles.add(collectible)
        all_sprites.add(collectible)

    elif level == 3:
        # Create the boss and add it to all_sprites
        global boss
        boss = BossEnemy(600, SCREEN_HEIGHT - 150)  # Set the starting position of the boss
        all_sprites.add(boss)  # Add the boss to the all_sprites group
        # Optionally, add some regular enemies in level 3
        for i in range(2):
            enemy = Enemy(random.randint(400, 800), SCREEN_HEIGHT - 150)
            enemies.add(enemy)
            all_sprites.add(enemy)

    all_sprites.add(player)  # Always add the player to all_sprites

def complete_level(level):
    # Show the "Level Complete" message
    level_transition_message(screen, f"Level {level} Complete", font)
    pygame.time.delay(1000)

    # Move to the next level, or show the game end menu if all levels are completed
    if level < 3:
        global current_level
        current_level += 1  # Move to the next level
        setup_level(current_level)
    else:
        game_end_menu()  # Show the end menu instead of exiting the game

def game_won_message():
    # Display the "You Win!" message when the final level is completed
    screen.fill(BLACK)
    text = font.render("You Win! Game Complete!", True, WHITE)
    screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 - text.get_height() // 2))
    pygame.display.flip()
    pygame.time.delay(3000)  # Display the message for 3 seconds
    pygame.quit()
    sys.exit()

# In the main game loop, update enemies in every level
def main():
    global player
    player = Player()
    global bullets
    bullets = pygame.sprite.Group()
    global enemies
    enemies = pygame.sprite.Group()
    global collectibles
    collectibles = pygame.sprite.Group()
    global all_sprites
    all_sprites = pygame.sprite.Group()

    global current_level
    current_level = 1
    setup_level(current_level)

    camera = Camera(background_width, SCREEN_HEIGHT)

    run = True
    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player.move(-player.speed, 0)
        if keys[pygame.K_RIGHT]:
            player.move(player.speed, 0)
        if keys[pygame.K_SPACE]:
            player.shoot()
        if keys[pygame.K_UP]:
            player.jump()

        # Update player and bullets
        player.update()
        camera.update(player)
        bullets.update()

        # Update all enemies
        enemies.update(player)

        # Update the boss if it's level 3
        if current_level == 3 and 'boss' in globals():
            boss.update(player)

        # Collision detection: check if bullets hit enemies
        bullet_enemy_collisions = pygame.sprite.groupcollide(bullets, enemies, True, False)
        for bullet, hit_enemies in bullet_enemy_collisions.items():
            for enemy in hit_enemies:
                enemy.take_damage(bullet.damage)

        # Collision detection: check if bullets hit the boss
        if current_level == 3 and 'boss' in globals():
            bullet_boss_collisions = pygame.sprite.spritecollide(boss, bullets, True)
            for bullet in bullet_boss_collisions:
                boss.take_damage(bullet.damage)

        # Check if all enemies are defeated or boss in level 3
        if current_level == 3 and 'boss' in globals() and boss.health <= 0:
            complete_level(current_level)
        elif len(enemies) == 0 and current_level != 3:  # If no enemies remain in the level (except boss level)
            complete_level(current_level)

        # Draw the background and all sprites
        rel_x = camera.camera.x % background_width
        screen.blit(background_image, (rel_x - background_width, 0))
        if rel_x < SCREEN_WIDTH:
            screen.blit(background_image, (rel_x, 0))

        all_sprites.draw(screen)
        bullets.draw(screen)
        collectibles.draw(screen)

        player.draw_health_bar()

        for enemy in enemies:
            enemy.draw_health_bar()

        if current_level == 3 and 'boss' in globals():
            boss.draw_health_bar()

        pygame.display.flip()

    pygame.quit()

def game_end_menu():
    screen.fill(BLACK)
    end_text = font.render("Congratulations! You've completed the game!", True, WHITE)
    replay_text = font.render("Press R to Replay or Q to Quit", True, WHITE)
    screen.blit(end_text, (SCREEN_WIDTH // 2 - end_text.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
    screen.blit(replay_text, (SCREEN_WIDTH // 2 - replay_text.get_width() // 2, SCREEN_HEIGHT // 2 + 10))
    pygame.display.flip()

    waiting = True
    while waiting:
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
