import torch

import environment as env
from environment.joueur import Joueur
import environment.ressources.maps as maps
from utils import olds_architecure, dqn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_name = 'fourth_generation'
model_name = 'link_ad_6layer_shield.gen7'
modele_hero = dqn.DQN

model_name_adversaire = 'link_ad_6layer.gen5'
modele_adv = dqn.DQN

avec_adversaire= True
show_runs = True
random_maps = True
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state_hero observations
obs_hero, obs_adv, info = env.reset()
state_hero_a, state_hero_b = obs_hero
n_observations_a = len(state_hero_a)

n_observations_b = len(state_hero_b)

player_1 = Joueur('./models/' + model_name, n_observations_a, n_observations_b+2, n_actions, device, modele_hero)
player_2 = Joueur('./models/' + model_name_adversaire, n_observations_a, n_observations_b+2, n_actions-1, device, modele_adv)

l_score_atq = []
l_score_dfs = []

for i in range(len(maps.echantillon_carte)):
    somme_score_atq = 0
    somme_score_dfs = 0
    num_iteration = 10

    for k in range(num_iteration): # on moyenne sur 6 scores

        if random_maps:
            observation_hero, observation_adversaire, info = env.reset()
        else:
            observation_hero, observation_adversaire, info = env.reset(choose_map=i)
        
        state_hero_a, state_hero_b = observation_hero
        state_adversaire_a, state_adversaire_b = observation_adversaire

        state_hero_a = torch.tensor(state_hero_a, dtype=torch.float32, device=device).unsqueeze(0)
        state_hero_b = torch.tensor(state_hero_b, dtype=torch.float32, device=device).unsqueeze(0)
        state_adversaire_a = torch.tensor(state_adversaire_a, dtype=torch.float32, device=device).unsqueeze(0)
        state_adversaire_b = torch.tensor(state_adversaire_b, dtype=torch.float32, device=device).unsqueeze(0)
        
        t=-1
        terminated = False
        while True:
            t+=1
            action_a, action_b = player_1.action(state_hero_a, state_hero_b, link_ab=True)
            action_adversaire_a, action_adversaire_b = player_2.action(state_adversaire_a, state_adversaire_b, link_ab=True)

            if avec_adversaire:
                observation_hero, observation_adversaire, reward_atq, reward_dfs, terminated, _ = env.step((action_a, action_b), (action_adversaire_a, action_adversaire_b))
            else:
                observation_hero, observation_adversaire, reward_atq, reward_dfs, terminated, _ = env.step((action_a, action_b), (0, 0))
            
            if terminated:
                if show_runs:
                    env.afficher()
                
                n_cp_hero, n_cp_adversaire = env.get_cp()

                sc_atq = n_cp_hero
                sc_dfs = -n_cp_adversaire
                # print(sc_atq, sc_dfs)
                somme_score_atq += sc_atq
                somme_score_dfs += sc_dfs
                break
            else: 
                nxt_st_a, nxt_st_b = observation_hero
                state_hero_a = torch.tensor(nxt_st_a, dtype=torch.float32, device=device).unsqueeze(0)
                state_hero_b = torch.tensor(nxt_st_b, dtype=torch.float32, device=device).unsqueeze(0)

                stt_adv_a, stt_adv_b = observation_adversaire
                state_adversaire_a = torch.tensor(stt_adv_a, dtype=torch.float32, device=device).unsqueeze(0)
                state_adversaire_b = torch.tensor(stt_adv_b, dtype=torch.float32, device=device).unsqueeze(0)

    if not random_maps:
        print("attaque", i, somme_score_atq/num_iteration)
        print("defense", i, somme_score_dfs/num_iteration)
        l_score_atq.append(somme_score_atq/num_iteration)
        l_score_dfs.append(somme_score_dfs/num_iteration)

if not random_maps:
    print(l_score_atq)
    print(l_score_dfs)