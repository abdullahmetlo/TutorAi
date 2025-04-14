import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from enum import Enum
import os
import joblib

# Load trained ML model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_cognitive_load_model.pkl")
try:
    ml_model = joblib.load(MODEL_PATH)
    print("✅ ML model loaded successfully!")
except FileNotFoundError:
    ml_model = None
    print("❌ ML model not found. Using heuristic fallback.")

# Define states and actions
class State(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2

class Action(Enum):
    PROVIDE_HINT = 0
    SHOW_EXAMPLE = 1
    GIVE_EXPLANATION = 2
    INTRODUCE_BREAK = 3

num_states = len(State)
num_actions = len(Action)

# Q-table initialized
q_table = np.zeros((num_states, num_actions))

# Heuristic-based cognitive load estimation
def estimate_cognitive_load(task_difficulty, error_rate, response_time):
    normalized_time = min(response_time / 60.0, 1.0)
    score = 0.4 * error_rate + 0.3 * normalized_time + 0.3 * task_difficulty
    if score < 0.3:
        return State.LOW
    elif score < 0.6:
        return State.MEDIUM
    else:
        return State.HIGH

# ML-based cognitive load estimation (uses trained model)
def ml_estimate_cognitive_load(task_difficulty, error_rate, response_time, avg_error_rate=0.5, avg_response_time=30.0):
    if ml_model is None:
        return estimate_cognitive_load(task_difficulty, error_rate, response_time)

    X = np.array([[task_difficulty, error_rate, response_time, avg_error_rate, avg_response_time]])
    prediction = ml_model.predict(X)[0]
    return State(prediction)  # 0 = LOW, 1 = MEDIUM, 2 = HIGH

# Reward Table
reward_table = {
    State.LOW: {
        Action.SHOW_EXAMPLE: 5,
        Action.GIVE_EXPLANATION: 3,
        Action.PROVIDE_HINT: -1,
        Action.INTRODUCE_BREAK: -2,
    },
    State.MEDIUM: {
        Action.PROVIDE_HINT: 3,
        Action.GIVE_EXPLANATION: 2,
        Action.INTRODUCE_BREAK: -2,
        Action.SHOW_EXAMPLE: 1,
    },
    State.HIGH: {
        Action.INTRODUCE_BREAK: 5,
        Action.PROVIDE_HINT: -2,
        Action.GIVE_EXPLANATION: 2,
        Action.SHOW_EXAMPLE: 1,
    }
}

def get_reward(state, action):
    return reward_table.get(state, {}).get(action, 0)

# Transition counts & probabilities
transition_counts = np.zeros((num_states, num_states, num_actions))
transition_probs = np.ones((num_states, num_states, num_actions)) / num_states  # Start with uniform

def update_transition_counts(current_state, next_state, action):
    transition_counts[current_state, next_state, action] += 1
    transition_probs[:, :, action] = transition_counts[:, :, action] / (
        transition_counts[:, :, action].sum(axis=1, keepdims=True) + 1e-6
    )

# Learning Parameters
alpha = 0.1
gamma = 0.9
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

# Training Loop
num_episodes = 1000
rewards_over_time = []

for episode in range(num_episodes):
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
    state = np.random.choice([s.value for s in State])
    total_reward = 0
    max_steps = 10

    for step in range(max_steps):
        if np.random.rand() < epsilon:
            action = np.random.choice([a.value for a in Action])
        else:
            action = np.argmax(q_table[state])

        reward = get_reward(State(state), Action(action))
        total_reward += reward

        probs = transition_probs[state, :, action]
        probs_sum = probs.sum()
        if probs_sum == 0 or np.isnan(probs_sum):
            probs = np.ones(num_states) / num_states
        else:
            probs = probs / probs_sum

        next_state = np.random.choice(num_states, p=probs)
        update_transition_counts(state, next_state, action)

        current_q = q_table[state, action]
        max_next_q = np.max(q_table[next_state])
        td_target = reward + gamma * max_next_q
        q_table[state, action] += alpha * (td_target - current_q)

        state = next_state

    rewards_over_time.append(total_reward)
    if episode % 100 == 0:
        print(f"Episode {episode}: Total reward = {total_reward}")

np.save("transition_probs.npy", transition_probs)
np.save("trained_q_table.npy", q_table)
print("✅ Q-table saved successfully!")

plt.plot(rewards_over_time)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress")
plt.savefig("training_progress.png")
plt.close()

print("Trained Q-table:")
print(q_table)

# Django Integration
def get_rl_recommendation(user_data, q_table):
    state = estimate_cognitive_load(
        user_data["task_difficulty"],
        user_data["error_rate"],
        user_data["response_time"]
    ).value

    action = np.argmax(q_table[state])
    actions_map = {
        0: "Provide Hint",
        1: "Show Example",
        2: "Give Explanation",
        3: "Introduce Break"
    }
    content_map = {
        "Provide Hint": "Here’s a useful hint for the problem...",
        "Show Example": "Check out this example for better understanding...",
        "Give Explanation": "Detailed explanation of the concept...",
        "Introduce Break": "Take a short break to refresh your mind!"
    }

    rl_decision = actions_map.get(action, "Unknown Action")
    return {"decision": rl_decision, "next_content": content_map.get(rl_decision, "No content available")}

def load_q_table():
    try:
        q_table = np.load("trained_q_table.npy")
        print("✅ Loaded existing Q-table from file!")
    except FileNotFoundError:
        print("❌ No Q-table found. Initializing a new one.")
        q_table = np.zeros((3, 4))
        np.save("trained_q_table.npy", q_table)
    return q_table
