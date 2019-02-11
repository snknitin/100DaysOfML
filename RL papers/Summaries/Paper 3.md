#  Dueling Network Architectures for Deep Reinforcement Learning


Most of the approaches for RL use standard neural networks, such as convolutional networks, MLPs, LSTMs and autoencoders.

dueling network represents two separate estimators: one for the state value function and one for the state-dependent action advantage function. The main benefit of this factoring is to generalize learning across actions without imposing any change to the underlying reinforcement learning algorithm.              
This architecture leads to better policy evaluation in the presence of many similar-valued actions.
             
The dueling architecture consists of two streams that represent the value and advantage functions, while sharing a common  
convolutional feature learning module. The two streams are combined via a special aggregating layer to produce an estimate of the state-action value function Q

The dueling network automatically produces separate estimates of the state value function and advantage function, without any extra supervision.Intuitively, the dueling architecture can learn which states are (or are not) valuable, without having to learn the effect of each action for each state. This is particularly useful in states where its actions do not affect the environment in any relevant way.   

**Advantage updating was shown to converge faster than Q-learning in simple continuous time domains**.The dueling architecture can more quickly identify the correct action during policy evaluation as redundant or similar actions are added to the learning problem.The value functions are high dimensional objects. To approximate them, we can use a deep Q-network: Q(s; a; θ) with parameters θ. To estimate this network, we optimize the following sequence of loss functions at iteration i: 

			Li(θi) = E[ yDQN^2 − Q(s; a; θi)^2]
            
			yDQN = r + γ maxQ(s0; a0; θ−)
            
θ− represents the parameters of a fixed and separate target network.
             
We could attempt to use standard Qlearning to learn the parameters of the network Q(s; a; θ)
online. However, this estimator performs poorly in practice. A key innovation in (Mnih et al., 2015) was to freeze
the parameters of the target network Q(s0; a0; θ−) for a
fixed number of iterations while updating the online network Q(s; a; θi) by gradient descent.
             
This approach is model free in the sense that the states and
rewards are produced by the environment. It is also offpolicy because these states and rewards are obtained with
a behavior policy (epsilon greedy in DQN) different from
the online policy that is being learned.
             
Another key ingredient behind the success of DQN is experience replay
             
During learning, the agent accumulates a dataset Dt = fe1; e2; : : : ; etg
of experiences et = (st; at; rt; st+1) from many episodes.
When training the Q-network, instead only using the
current experience as prescribed by standard temporaldifference learning, the network is trained by sampling
mini-batches of experiences from D uniformly at random.
             
the advantage function, relating the value and Q functions:

			Aπ(s; a) = Qπ(s; a) − V π(s)
             
			Note that Ea∼π(s) [Aπ(s; a)] = 0
             
Intuitively, the value function V measures the how good it is to be in a particular state s. The Q function, however, measures the the value of choosing a particular action when in this state. The advantage function subtracts the value of the state from the Q function to obtain a relative measure of the importance of each action.

In Q-learning and DQN, the max operator uses the same values to both select and evaluate an action. This can therefore lead to overoptimistic value estimates (van Hasselt, 2010). To mitigate this problem, DDQN uses the following target:

		y_DDQN = r + γQ(s0; arg max_a'Q(s0; a0; θi); θ−):
             
Since the output of the dueling network is a Q function, it can be trained with the many existing algorithms, such as DDQN and SARSA.  

			V π(s) = Ea∼π(s) [Qπ(s; a)], it follows that E a∼π(s) [Aπ(s; a)] = 0.

Moreover, for a deterministic policy, a∗ = arg maxa02A Q(s; a0), it follows that Q(s; a) = V (s) and hence A(s; a) = 0.
             
key idea was to increase the replay probability of experience tuples that have a high expected learning progress (as measured via the proxy of absolute TD-error). This led to both faster learning and to better final policy quality.


The lower layers of the dueling network are convolutional as in the original DQNs. Instead of following the convolutional layers with a single sequence of fully connected layers, we instead use two sequences (or streams) of fully connected layers. The streams are constructed such that they have they have the capability of providing separate estimates of the value and advantage funct ons.
             
To evaluate the learned Q values, we choose a simple environment where the exact Qπ(s; a) values can be computed separately for all (s; a) 2 S × A. This environment, which we call the corridor is composed of three connected corridors 

The agent starts from the bottom left corner of the environment and must move to the top right to get the largest reward. A total of 5 actions are available: go up, down, left, right and no-op. We also have the freedom of adding an arbitrary number of no-op actions. In our setup, the two vertical sections both have 10 states while the horizontal section has 50.No Comments.   

For example, an agent that achieves 2% human performance should not be interpreted as two times better when the baseline agent achieves 1% human performance. We also chose not to measure performance in terms of percentage of human performance alone because a tiny difference relative to the baseline on some games can translate into hundreds of percent in human performance difference.      

Saliency maps. To better understand the roles of the value and the advantage streams, we compute saliency maps (Simonyan et al., 2013). More specifically, to visualize the salient part of the image as seen by the value stream, we compute the absolute value of the Jacobian of Vb with respect to the input frames. Both quantities are of the same dimensionality as the input frames and therefore can be visualized easily alongside the input frames.


The advantage of the dueling architecture lies partly in its ability to learn the state-value function efficiently. With every update of the Q values in the dueling architecture, the value stream V is updated – this contrasts with the updates in a single-stream architecture where only the value for one of the actions is updated, the values for all other actions remain untouched. This more frequent updating of the value stream in our approach allocates more resources to V , and thus allows for better approximation of the state values, which in turn need to be accurate for temporaldifference-based methods like Q-learning to work. Furthermore, the differences between Q-values for a given state are often very small relative to the magnitude of Q.This difference in scales can lead to small amounts of noise in the updates can lead to reorderings of the actions, and thus make the nearly greedy policy switch abruptly. 