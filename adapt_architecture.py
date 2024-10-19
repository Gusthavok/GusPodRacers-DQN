import torch
from utils.dqn import DQN

old_weights = torch.load('./models/link_ad_6layer.gen5_atq', map_location='cpu')

n_observations = 20
n_actions = 16


new_model = DQN(n_observations, n_actions)

model_dict = new_model.state_dict()
pretrained_dict = {k: v for k, v in old_weights.items() if k in model_dict and v.size() == model_dict[k].size()}

model_dict.update(pretrained_dict)


torch.save(model_dict, './models/link_ad_6layer_shield.gen5_atq')
