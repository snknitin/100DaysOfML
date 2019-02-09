# [2] Deep Recurrent Q-Learning for Partially Observable MDPs


Deep Reinforcement Learning has yielded proficient controllers for complex tasks. However, these controllers have limited memory and rely on being able to perceive the complete game screen at each decision point. The effects of adding recurrency to a Deep Q-Network (DQN) by replacing the first post-convolutional fully-connected layer with a recurrent LSTM. The resulting Deep Recurrent Q-Network (DRQN), although capable of seeing only a single frame at each timestep, successfully integrates information through time and replicates DQN’s performance on standard Atari games and partially observed equivalents featuring flickering game screens. Given the same length of history, recurrency is a viable alternative to stacking a history of frames in the DQN’s input layer and while recurrency confers no systematic advantage when learning to play the game, the recurrent net can better adapt at evaluation time if the quality of observations changes 

Put differently, any game that requires a memory of more than four frames will appear non-Markovian because the future game states (and rewards) depend on more than just DQN’s current input. Instead of a Markov Decision Process (MDP), the game becomes a Partially-Observable Markov Decision Process (POMDP). Deep Recurrent Q-Network (DRQN), a combination of a Long Short Term Memory (LSTM) (Hochreiter and Schmidhuber 1997) and a Deep Q-Network.

A Partially Observable Markov Decision Process (POMDP) better captures the dynamics of many realworld environments by explicitly acknowledging that the sensations received by the agent are only partial glimpses of the underlying system state. Formally a POMDP can be described as a 6-tuple (S; A; P; R; Ω; O). S; A; P; R are the states, actions, transitions, and rewards as before, except now the agent is no longer privy to the true system state and instead receives an observation o 2 Ω. This observation is generated from the underlying system state according to the probability distribution o ∼ O(s).

Many challenging domains such as Atari games feature far too many unique states to maintain a separate estimate for each S × A. In the case of Deep QLearning, the model is a neural network parameterized by weights and biases collectively denoted as θ. Q-values are estimated online by querying the output nodes of the network after performing a forward pass given a state input. Such Q-values are denoted Q(s; ajθ). Instead of updating individual Q-values, updates are now made to the parameters of the network to minimize a differentiable loss function . Stated differently, recurrent deep Q-networks can better approximate actual Q-values from sequences of observations, leading to better policies in partially observed environments. Since jθj jS × Aj, the neural network model naturally generalizes beyond the states and actions it has been trained on. However, because the same network is generating the next state target Q-values that are used in updating its current Q-values, such updates can oscillate or diverge.  

Three techniques to restore learning stability: 

* First, experiences et = (st; at; rt; st+1) are recorded in a replay memory D and then sampled uniformly at training time. 

* Second, a separate, target network Q^ provides update targets to the main network, decoupling the feedback resulting from the network generating its own targets. Q^ is identical to the main network except its parameters θ− are updated to match θ every 10,000 iterations.D may require certain parameters start changing again after having reached a seeming fixed point. A higher Q-value indicates an action a is judged to yield better long-term results in a state s   

* Finally, an adaptive learning rate method such as RMSProp (Tieleman and Hinton 2012) or ADADELTA (Zeiler 2012) maintains a per-parameter learning rate α, and adjusts α according to the history of gradient updates to that parameter. This step serves to compensate for the lack of a fixed training dataset; the ever-changing nature 

Random updates better adhere to the policy of randomly sampling experience, but, as a consequence, the LSTM’s hidden state must be zeroed at the start of each update. Zeroing the hidden state makes it harder for the LSTM to learn functions that span longer time scales than the number of timesteps reached by back propagation through time. 


Since the explored games are fully observable given four input frames, we need a way to introduce partial observability without reducing the number of input frames given to DQN 

To address this problem, we introduce the Flickering Pong POMDP - a modification to the classic game of Pong such that at each timestep, the screen is either fully revealed or fully obscured with probability p = 0:5. Obscuring frames in this manner probabilistically induces an incomplete memory of observations needed for Pong to become a POMDP 
Updating a recurrent, convolutional network requires each backward pass to contain many time-steps of game screens and target values. Additionally, the LSTM’s initial hidden state may either be zeroed or carried forward from its previous values. 

We consider two types of updates:


* **Bootstrapped Sequential Updates:** Episodes are selected randomly from the replay memory and updates begin at the beginning of the episode and proceed forward through time to the conclusion of the episode. The targets at each timestep are generated from the target Q-network, Q^. The RNN’s hidden state is carried forward throughout the episode. Since half of the frames are obscured in expectation, a successful player must be robust to the possibility of several potentially contiguous obscured inputs. 

* **Bootstrapped Random Updates:** Episodes are selected randomly from the replay memory and updates begin at random points in the episode and proceed for only unroll iterations timesteps (e.g. one backward call). The targets at each timestep are generated from the target Q-network, Q^. The RNN’s initial state is zeroed at the start of the update. 

Sequential updates have the advantage of carrying the LSTM’s hidden state forward from the beginning of the episode. However, by sampling experiences sequentially for a full episode, they violate DQN’s random sampling policy 


DRQN is trained using backpropagation through time for the last ten timesteps. Thus both the non-recurrent 10-frame DQN and the recurrent 1-frame DRQN have access to the same history of game screens.3 Thus, when dealing with partial observability, a choice exists between using a nonrecurrent deep network with a long history of observations or using a recurrent network trained with a single observation at each timestep. 

Given the last four frames of input, all of these games are MDPs rather than POMDPs. Thus there is no reason to expect DRQN to outperform DQN. On average, DRQN does roughly as well DQN. By jointly training convolutional and LSTM layers we are able to learn directly from pixels and do not require hand-engineered features 

**Can a recurrent network be trained on a standard MDP and then generalize to a POMDP at evaluation time?**

While both algorithms incur significant performance decreases on account of the missing information, DRQN captures more of its previous performance than DQN across all levels of flickering. We conclude that recurrent controllers have a certain degree of robustness against missing information, even trained with full state information  Real-world tasks often feature incomplete and noisy state information, resulting from partial observability. We modify DQN to handle the noisy observations characteristic of POMDPs by combining a Long Short Term Memory with a Deep Q-Network. The resulting Deep Recurrent Q-Network (DRQN), despite seeing only a single frame at each step, is still capable integrating information across frames to detect relevant information such as velocity of on-screen objects 