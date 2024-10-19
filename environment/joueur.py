import torch
from utils.dqn import DQN
import environment.action_space as action_space
import environment as env

translate_action = [
    (angle, vitesse) for vitesse in action_space.vitesses for angle in action_space.angles
]

class Joueur:
    def __init__(self, joueur_path, n_observations_a, n_observations_b, n_actions, device, classe):
        self.n_actions = n_actions

        self.reseau_a = classe(n_observations=n_observations_a, n_actions=n_actions).to(device)
        self.reseau_a.load_state_dict(torch.load(joueur_path + '_atq', map_location=device))
        self.reseau_a.eval()

        self.reseau_b = classe(n_observations=n_observations_b, n_actions=n_actions).to(device)
        self.reseau_b.load_state_dict(torch.load(joueur_path + '_dfs', map_location=device))
        self.reseau_b.eval()

        self.device = device

    def action(self, state_a, state_b, link_ab = True):
        attaque_ind = self.reseau_a(state_a).max(1).indices.view(1, 1).item()

        if link_ab:
            ang1, pow1, _a, _b =  env.get_action((attaque_ind, 0))
            if pow1 == 'SHIELD':
                pow1 = 0
            elif pow1 == 'BOOST':
                pow1 = 650
            state_b = torch.cat([state_b] + [torch.tensor([[entree]]).to(self.device) for entree in (ang1, pow1)], dim=1)
            
        defense_ind = self.reseau_b(state_b).max(1).indices.view(1, 1).item()

        return attaque_ind, defense_ind
