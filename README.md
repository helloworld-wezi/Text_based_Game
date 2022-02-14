# Text_based_Game
Reinforcement Learning

## Intro
In this project, we address the task of learning control policies for text-based games using reinforcement learning. In these games, all interactions between players and the virtual world are through text. The current world state is described by elaborate text, and the underlying state is not directly observable. Players read descriptions of the state and respond with natural language commands to take actions.

For this project we will conduct experiments on a small Home World, which mimic the environment of a typical house.The world consists of a few rooms, and each room contains a representative object that the player can interact with. For instance, the kitchen has an apple that the player can eat. The goal of the player is to finish some quest. An example of a quest given to the player in text is You are hungry now . To complete this quest, the player has to navigate through the house to reach the kitchen and eat the apple. In this game, the room is hidden from the player, who only receives a description of the underlying room. At each step, the player read the text describing the current room and the quest, and respond with some command (e.g., eat apple ). The player then receives some reward that depends on the state and his/her command.

In order to design an autonomous game player, we will employ a reinforcement learning framework to learn command policies using game rewards as feedback. Since the state observable to the player is described in text, we have to choose a mechanism that maps text descriptions into vector representations. A naive approach is to create a map that assigns a unique index for each text description. However, such approach becomes difficult to implement when the number of textual state descriptions are huge. An alternative method is to use a bag-of-words representation derived from the text description.

Tasks:
1. Implement the tabular Q-learning algorithm for a simple setting where each text description is associated with a unique index.
2. Implement the Q-learning algorithm with linear approximation architecture, using bag-of-words representation for textual state description.
3. Implement a deep Q-network.
4. Use the Q-learning algorithms on the Home World game.

## Home World Game

![image](https://user-images.githubusercontent.com/87055709/153843604-b8da95c9-0554-4f25-914c-5cb59e8e4396.png)
We will conduct experiments on a small Home World, which mimic the environment of a typical house. The world consists of four rooms- a living room, a bed room, a kitchen and a garden with connecting pathways (illustrated in figure below). Transitions between the rooms are **deterministic**. Each room contains a representative object that the player can interact with. For instance, the living room has a **TV** that the player can **watch** , and the kitchen has an **apple** that the player can **eat**. Each room has several descriptions, invoked randomly on each visit by the player.

![image](https://user-images.githubusercontent.com/87055709/153843779-c161b5db-d243-4c63-892f-a8195c3130dc.png)
![image](https://user-images.githubusercontent.com/87055709/153843846-8e0f9723-e9ab-45b2-be9d-33dbc1ab62a2.png)
![image](https://user-images.githubusercontent.com/87055709/153843906-73848b3f-5fd0-4d09-9c18-36560d922928.png)
![image](https://user-images.githubusercontent.com/87055709/153843947-db510571-bcfa-45f5-ae4d-f67be4c3b82c.png)

Using the defined optimal expected reward achievable, it can be computed that each episode the expected optimal reward is 0.55375.

## Q-learning Algorithm
In this section, we will implement the Q-learning algorithm, which is a model-free algorithm used to learn an optimal Q-function. In the tabular setting, the algorithm maintains the Q-value for all possible state-action pairs. Starting from a random Q-function, the agent continuously collects experiences (_s_, _c_, _R_(_s_, _c_), _s'_) and updates its Q-function.

![image](https://user-images.githubusercontent.com/87055709/153844590-fc536a20-17aa-4ee3-95b4-6cbc10e67b67.png)

### Epsilon-greedy exploration
![image](https://user-images.githubusercontent.com/87055709/153844757-3daa11e9-deaf-4fc2-9a19-6943e21d1e17.png)

## Tabular Q-learning for Home World game
In this section we will consider a simple approach that assigns a unique index for each text description. In particular, we will build two dictionaries:
- dict_room_desc that takes the room description text as the key and returns a unique scalar index
- dict_quest_desc that takes the quest description text as the key and returns a unique scalar index.

![image](https://user-images.githubusercontent.com/87055709/153845137-66657beb-2910-4f7e-909b-bbd172661b24.png)

### Performance
By initializing:
- Q at zero
- Number of runs 10
- Number of episodes for training at each epoch 25
- Number of episodes for testing 50
- Discount factor (gamma) 0.5
- Training epsilon-greedy 0.5
- Testing epsion-greedy 0.05
- Learning rate (alpha) 0.1 (too large=learning is too unstable but the faster the convergence, too small=learning is too slow to converge)

The number of epoch in which the learning algorithm converges is 13, meanwhile the average episodic rewards of the Q-learning algorithm when it converges is 0.5058.

![Fig_1_Agent_Tabular](https://user-images.githubusercontent.com/87055709/153847143-bc7dcd06-1750-4696-91b6-c01e8bff8178.png)


## Q-learning with linear function approximation

### First Approach

![image](https://user-images.githubusercontent.com/87055709/153845906-4262072f-cf89-49e2-a990-2958bbbf4851.png)
The approach as explained above won't work since the feature engineering is not sufficient to learn a good approximation of the optimal Q-function.

### Second Approach
![image](https://user-images.githubusercontent.com/87055709/153846154-39c7774c-be23-4313-92e5-bd3b1079c462.png)

#### Computing theta update rule
![image](https://user-images.githubusercontent.com/87055709/153846321-d08a9e7e-b5be-4894-b8b2-208673f94b38.png)
![image](https://user-images.githubusercontent.com/87055709/153846349-3da3ee6a-9255-477c-a86d-7e4e3469eca8.png)
![image](https://user-images.githubusercontent.com/87055709/153846505-7d1c5ea7-0985-4161-8346-eaf73ebfe3a3.png)
![image](https://user-images.githubusercontent.com/87055709/153846539-4b8ce084-3013-4637-9da9-ed4a32af8dc3.png)

Using the hyperparameters:
- Number of runs 5,
- Number of episodes for training at each epoch 25,
- Number of episodes for testing 50,
- Discount factor (gamma) 0.5,
- Training epsilon-greedy 0.5,
- Testing epsion-greedy 0.05, and
- Learning rate (alpha) 0.01,
the average episodic rewards of the Q-learning algorithm when it converges is 0.3862.

![Fig_2_Agent_Linear](https://user-images.githubusercontent.com/87055709/153847953-a29e4987-750e-429c-b37f-56853f307ef2.png)


## Deep Q-network (DQN)
As seen in the previous part, a linear model is not able to correctly approximate the Q-function for our simple task. In this part, we will approximate _Q_(_s_, _c_) with a neural network. We will be using a DQN that takes the state representation (bag-of-words) and outputs the predicted Q values for the different "actions" and "objects".

The hyperparameter values are as follows:
- Number of runs 10,
- Number of episodes for training at each epoch 25,
- Number of episodes for testing 50,
- Discount factor (gamma) 0.5,
- Training epsilon-greedy 0.5,
- Testing epsion-greedy 0.05, and
- Learning rate (alpha) 0.1,

The setup of the neural nets is as follows:
- One fully connected layer with size 100 (state encoder)
- One fully connected layer with size equals to action dimension (4) to get the action index
- One fully connected layer with size equals to object dimension (4) to get the object index

Using the DQN, the average episodic rewards of the Q-learning algorithm when it converges is 0.48.

![Fig_3_Agent_DQN](https://user-images.githubusercontent.com/87055709/153849421-b5959d27-f439-42ea-b83a-4f4b968555ed.png)
