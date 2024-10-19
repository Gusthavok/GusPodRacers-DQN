# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer


# attaquant :
BATCH_SIZE_a = 20000
GAMMA_a = 0.9
EPS_START_a = 0.0
EPS_END_a = 0.0
EPS_DECAY_a = 45000 # 150 runs, 300 ticks par run
TAU_a = 0.005
LR_a = 1e-4

# defenseur :
BATCH_SIZE_b = 16384
GAMMA_b = 0.95
EPS_START_b = 0.9
EPS_END_b = 0.05
EPS_DECAY_b = 45000
TAU_b = 0.005
LR_b = 2e-5 