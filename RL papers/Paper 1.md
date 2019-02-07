
The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards.Firstly, most successful deep learning applications to date have required large amounts of handlabelled training data. RL algorithms, on the other hand, must be able to learn from a scalar reward signal that is frequently sparse, noisy and delayed. The delay between actions and resulting rewards, which can be thousands of timesteps long, seems particularly daunting when compared to the direct association between inputs and targets found in supervised learning 

most deep learning algorithms assume the data samples to be independent, while in reinforcement learning one typically encounters sequences of highly correlated states. In RL the data distribution changes as the algorithm learns new behaviours, which can be problematic for deep learning methods that assume a fixed underlying distribution

To alleviate the problems of correlated data and non-stationary distributions, we use an experience replay mechanism which randomly samples previous transitions, and thereby smooths the training distribution over many past behaviors. 

Our goal is to create a single neural network agent that is able to successfully learn to play as many of the games as possible


The network was not provided with any game-specific information or hand-designed visual features, and was not privy to the internal state of the emulator; it learned from nothing but the video input, the reward and terminal signals, and the set of possible actions—just as a human player would. 

Furthermore the network architecture and all hyperparameters used for training were kept constant across the games.


At each time-step the agent selects an action at from the set of legal game actions, A = f1; : : : ; Kg. The action is passed to the emulator and modifies its internal state and the game score. In general E may be stochastic. The emulator’s internal state is not observed by the agent; instead it observes an image xt 2 Rd from the emulator, which is a vector of raw pixel values representing the current screen. In addition it receives a reward rt representing the change in game score. Note that in general the game score may depend on the whole prior sequence of actions and observations; feedback about an action may only be received after many thousands of time-steps have elapsed 

it is impossible to fully understand the current situation from only the current screen xt. We therefore consider sequences of actions and observations, st = x1; a1; x2; :::; at−1; xt, and learn game strategies that depend upon these sequences. All sequences in the emulator are assumed to terminate in a finite number of time-steps. This formalism gives rise to a large but finite Markov decision process (MDP) in which each sequence is a distinct state 

The goal of the agent is to interact with the emulator by selecting actions in a way that maximises future rewards.


The optimal action-value function obeys an important identity known as the Bellman equation. This is based on the following intuition: if the optimal value Q∗(s0; a0) of the sequence s0 at the next time-step was known for all possible actions a0, then the optimal strategy is to select the action  

The basic idea behind many reinforcement learning algorithms is to estimate the actionvalue function, by using the Bellman equation as an iterative update, Qi+1(s; a) = E [r + γ maxa0 Qi(s0; a0)js; a]. Such value iteration algorithms converge to the optimal actionvalue function, Qi ! Q as i ! 1 


totally impractical, because the action-value function is estimated separately for each sequence, without any generalisation. Instead, it is common to use a function approximator to estimate the action-value function, Q(s; a; θ) ≈ Q(s; a). ρ(s; a) is a probability distribution over sequences s and actions a that we refer to as the behaviour distribution.


The parameters from the previous iteration θi−1 are held fixed when optimising the loss function Li (θi).


This algorithm is model-free: it solves the reinforcement learning task directly using samples from the emulator E, without explicitly constructing an estimate of E. It is also off-policy: it learns about the greedy strategy a = maxa Q(s; a; θ), while following a behaviour distribution that ensures adequate exploration of the state space.


we utilize a technique known as experience replay [13] where we store the agent’s experiences at each time-step, et = (st; at; rt; st+1) in a data-set D = e1; :::; eN, pooled over many episodes into a replay memory 

we apply Q-learning updates, or minibatch updates, to samples of experience, e ∼ D, drawn at random from the pool of stored samples.

This approach has several advantages over standard online Q-learning [23]. First, each step of experience is potentially used in many weight updates, which allows for greater data efficiency 

Second, learning directly from consecutive samples is inefficient, due to the strong correlations between the samples; randomizing the samples breaks these correlations and therefore reduces the variance of the updates. Third, when learning on-policy the current parameters determine the next data sample that the parameters are trained on. For example, if the maximizing action is to move left then the training samples will be dominated by samples from the left-hand side; if the maximizing action then switches to the right then the training distribution will also switch. It is easy to see how unwanted feedback loops may arise and the parameters could get stuck in a poor local minimum, or even diverge catastrophically. By using experience replay the behavior distribution is averaged over many of its previous states, smoothing out learning and avoiding oscillations or divergence in the parameters. Note that when learning by experience replay, it is necessary to learn off-policy (because our current parameters are different to those used to generate the sample), which motivates the choice of Q-learning. 

our algorithm only stores the last N experience tuples in the replay memory, and samples uniformly at random from D when performing updates. This approach is in some respects limited since the memory buffer does not differentiate important transitions and always overwrites with recent transitions due to the finite memory size N. Similarly, the uniform sampling gives equal importance to all transitions in the replay memory. emphasize transitions from which we can learn the most, similar to prioritized sweeping 

While we evaluated our agents on the real and unmodified games, we made one change to the reward structure of the games during training only. Since the scale of scores varies greatly from game to game, we fixed all positive rewards to be 1 and all negative rewards to be −1, leaving 0 rewards unchanged. Clipping the rewards in this manner limits the scale of the error derivatives and makes it easier to use the same learning rate across multiple games. At the same time, it could affect the performance of our agent since it cannot differentiate between rewards of different magnitude 
we used the RMSProp algorithm with minibatches of size 32. The behavior policy during training was -greedy with annealed linearly from 1 to 0:1 over the first million frames, and fixed at 0:1 thereafter. We trained for a total of 10 million frames and used a replay memory of one million most recent frames 


Since our evaluation metric, is the total reward the agent collects in an episode or game averaged over a number of games, we periodically compute it during training. The average total reward metric tends to be very noisy because small changes to the weights of a policy can lead to large changes in the distribution of states the policy visits . 

Another, more stable, metric is the policy’s estimated action-value function Q, which provides an estimate of how much discounted reward the agent can obtain by following its policy from any given state 