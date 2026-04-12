import random
import pygame

# --- ENVIRONMENT ---
class GridWorld:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.agent_pos = (random.randint(2, width-3), random.randint(2, height-3))
        self.pit_positions = set() # Initialize empty, filled in reset
        self.goal_pos = (0,0)      # Initialize empty, filled in reset
        self.reset_map()

    def _generate_random_position(self):
        while True:
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            if (x, y) not in self.pit_positions and (x, y) != self.agent_pos:
                return (x, y)

    def _generate_random_pits(self, num_pits):
        all_possible = [(x, y) for x in range(1, self.width - 1) for y in range(1, self.height - 1)]
        if self.agent_pos in all_possible:
            all_possible.remove(self.agent_pos)
        pit_list = random.sample(all_possible, num_pits)
        return set(pit_list)

    def reset_map(self):
        # Triggers the random generation of the board
        self.pit_positions = set() # Clear old pits before making new ones
        self.agent_pos = (random.randint(2, self.width - 3), random.randint(2, self.height - 3))
        self.pit_positions = self._generate_random_pits(num_pits=15) # Increased pits for challenge
        self.goal_pos = self._generate_random_position()
        print("--- MAP RESET: New Goal and Pits Generated! ---")

    def reset_agent(self):
        # Resets just the agent to a random safe spot (used during training loop)
        while True:
            new_pos = (random.randint(1, self.width - 2), random.randint(1, self.height - 2))
            if new_pos not in self.pit_positions and new_pos != self.goal_pos:
                self.agent_pos = new_pos
                break
        return self.agent_pos

    def step(self, action):
        current_x, current_y = self.agent_pos
        new_x, new_y = current_x, current_y

        if action == 0: new_y -= 1   # Up
        elif action == 1: new_y += 1 # Down
        elif action == 2: new_x -= 1 # Left
        elif action == 3: new_x += 1 # Right

        # Check for collisions
        hit_wall = not (0 <= new_x < self.width and 0 <= new_y < self.height)
        
        if hit_wall or (new_x, new_y) in self.pit_positions:
            reward = -100 # Heavy penalty for pits/walls
            done = True   # End episode on death to learn faster
            self.agent_pos = (current_x, current_y) # Revert move
        elif (new_x, new_y) == self.goal_pos:
            reward = 100  # Reached the goal
            self.agent_pos = (new_x, new_y)
            done = True
        else:
            reward = -1   # Time penalty to encourage direct routes
            self.agent_pos = (new_x, new_y)
            done = False

        return self.agent_pos, reward, done

# --- RL AGENT ---
class QLearningAgent:
    def __init__(self, width, height):
        self.q_table = {} 
        self.learning_rate = 0.1  
        self.discount_factor = 0.95 
        self.epsilon = 1.0        

    def choose_action(self, state, exploit_only=False):
        # If exploit_only is True (used for visual display), bypass randomness
        if not exploit_only and random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            if state not in self.q_table or not self.q_table[state]:
                return random.randint(0, 3)
            return max(self.q_table[state], key=self.q_table[state].get)

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table.get(state, {}).get(action, 0.0)
        next_max_q = max(self.q_table.get(next_state, {}).values()) if self.q_table.get(next_state) else 0.0

        new_q = (1 - self.learning_rate) * current_q + \
                self.learning_rate * (reward + self.discount_factor * next_max_q)
        
        if state not in self.q_table:
            self.q_table[state] = {0:0.0, 1:0.0, 2:0.0, 3:0.0} # Initialize all actions
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.99)


# --- RAPID TRAINING FUNCTION ---
def train_on_current_map(env, agent, episodes=2000):
    """Trains the agent rapidly on the current static layout."""
    agent.epsilon = 1.0 # Reset exploration for new map
    for _ in range(episodes):
        current_state = env.reset_agent()
        done = False
        steps = 0
        while not done and steps < 100: # Max steps to prevent infinite loops
            action = agent.choose_action(current_state)
            next_state, reward, done = env.step(action)
            agent.learn(current_state, action, reward, next_state)
            current_state = next_state
            steps += 1
        agent.decay_epsilon()


# --- PYGAME VISUALIZATION & MAIN LOOP ---
def main():
    pygame.init()
    
    # Constants
    WIDTH, HEIGHT = 20, 20
    TILE_SIZE = 30
    SCREEN_WIDTH, SCREEN_HEIGHT = WIDTH * TILE_SIZE, HEIGHT * TILE_SIZE
    FPS = 8 # Speed of the agent's movement visually
    MAP_RESET_SECONDS = 15 # Change map every 15 seconds
    
    # Colors
    BG_COLOR = (40, 40, 40)
    GRID_COLOR = (60, 60, 60)
    AGENT_COLOR = (50, 150, 255)
    GOAL_COLOR = (50, 255, 50)
    PIT_COLOR = (255, 50, 50)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Q-Learning Automated GridWorld")
    clock = pygame.time.Clock()

    env = GridWorld(WIDTH, HEIGHT)
    agent = QLearningAgent(WIDTH, HEIGHT)

    # Initial Training
    print("Performing initial training...")
    train_on_current_map(env, agent, episodes=3000)
    env.reset_agent() # Put agent in a safe spot to start visualization

    # Set up the timer for map resets
    MAP_RESET_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(MAP_RESET_EVENT, MAP_RESET_SECONDS * 1000)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Every N seconds, this event fires
            if event.type == MAP_RESET_EVENT:
                env.reset_map()
                agent = QLearningAgent(WIDTH, HEIGHT) # Wipe old memory completely!
                print("Re-training on new map...")
                train_on_current_map(env, agent, episodes=3000)
                env.reset_agent()

        # --- AGENT AUTOMATION ---
        current_state = env.agent_pos
        # Force exploit=True so it uses its best learned path, no random wandering
        action = agent.choose_action(current_state, exploit_only=True) 
        next_state, reward, done = env.step(action)
        
        # If the agent reached the goal (or died by forcing itself into a pit), reset its position
        if done:
            env.reset_agent()

        # --- DRAWING ---
        screen.fill(BG_COLOR)

        # Draw Grid
        for x in range(0, SCREEN_WIDTH, TILE_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, TILE_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (0, y), (SCREEN_WIDTH, y))

        # Draw Pits
        for px, py in env.pit_positions:
            rect = pygame.Rect(px * TILE_SIZE, py * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, PIT_COLOR, rect)

        # Draw Goal
        gx, gy = env.goal_pos
        goal_rect = pygame.Rect(gx * TILE_SIZE, gy * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(screen, GOAL_COLOR, goal_rect)

        # Draw Agent
        ax, ay = env.agent_pos
        agent_rect = pygame.Rect(ax * TILE_SIZE, ay * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(screen, AGENT_COLOR, agent_rect)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()