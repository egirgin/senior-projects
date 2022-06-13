from mlp_with_tanh import Network
import gym
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

epsilon_decay = 0.999
gamma = 0.99
lr = 0.0001
num_episodes = 10000

def epsilon_greedy(output, e=0.1):
    if np.random.uniform() < e:
        return np.random.choice([0, 1])
    else:
        return np.argmax(output)

def run():
    e = 1.0

    reward_history = []

    net = Network(input_dim=4, hidden_dim=32, output_dim=2, num_layers=3)

    env = gym.make('CartPole-v1')
    for i_episode in range(num_episodes):
        observation = env.reset()
        cum_reward = 0
        for t in range(1000):
            # env.render()

            e *= epsilon_decay
            if e < 0.1:
                e = 0.1

            # calculate predicted action
            output = net.forward(observation)
            action = epsilon_greedy(output=output, e=e)
            pred = output[action]

            observation, reward, done, info = env.step(action)

            # save current intermediate values
            current_activations = net.activations.copy()
            current_z = net.z.copy()

            # needed for target value
            next_q = np.max(net.forward(observation))

            # reload current intermediate values
            net.activations = current_activations
            net.z = current_z

            if done:
                target = reward + gamma * 0

                error = np.zeros_like(output)

                error[action] = target - pred

                net.backward(error)

                net.update(lr=lr)

                reward_history.append(cum_reward)

                break
            else:
                target = reward + gamma * next_q # from bellman eqn

                error = np.zeros_like(output)

                error[action] = target - pred # from derivative of loss func

                net.backward(error)

                net.update(lr=lr)

                cum_reward += reward


    env.close()

    return reward_history


def moving_average(a, n=3):
    """
    from:
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



reward_hist = run()
plt.plot(moving_average(reward_hist, 50))
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.savefig("result.png")

