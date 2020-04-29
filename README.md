# openai-gym-implementations
Cartpole:
  - Simple implementation that uses a 4-tuple of 'weights' and the 4-tuple of the observation. If their dot product > 0, move right, else move left. The weights are randomly assigned, and the best weights of 100 trials are used in the final run. Works pretty well honestly!
  - Qlearning implementation which uses a dictionary to store the expected future reward of all state-action pairs. The space needs to be discretised to generate the dictionary. The expected future reward for a specific state-action pair is updated based on the actual reward received at that step, and the expected future reward of the state reached after taking that step. 

Pendulum:
- Qlearn implementation again. Action space also needed to be discretised. Implementation works fairly well, averaging -400 reward over 200 steps. Rendering shows the pendulum swinging fully around a few times before stabilising at the top. 
