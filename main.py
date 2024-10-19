import matplotlib.pyplot as plt

import torch
import torch.optim as optim

import environment as env
from environment.joueur import Joueur

from utils.dqn import DQN, ReplayMemory, select_action, optimize_model, soft_update
from utils.graph_train import plot_durations
from utils.learning_parameters import *
from utils import olds_architecure

####
# TO DO :
# -> pénalité si vitesse de l'attaquant trop faible
# -> pénalité si pod de défense trop loin
# 
# 
# 
# 
####

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = 'link_ad_6layer_shield.gen6'
model_name_reload = 'link_ad_6layer_shield.gen5'
only_atq = True
modele_hero = DQN

model_name_adversaire = 'link_ad_6layer.gen5'
modele_adv = DQN
is_adversaire = True

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state_hero observations
obs_hero, obs_adv, info = env.reset()
state_hero_a, state_hero_b = obs_hero
n_observations_a = len(state_hero_a)

policy_net_a = modele_hero(n_observations_a, n_actions).to(device)
target_net_a = modele_hero(n_observations_a, n_actions).to(device)
if model_name_reload != '':
    policy_net_a.load_state_dict(torch.load('./models/' + model_name_reload + '_atq', map_location=device))
    target_net_a.load_state_dict(torch.load('./models/' + model_name_reload + '_atq', map_location=device))
else:
    target_net_a.load_state_dict(policy_net_a.state_dict())
optimizer_a = optim.AdamW(policy_net_a.parameters(), lr=LR_a, amsgrad=True)
memory_atq = ReplayMemory(10000)

n_observations_b = len(state_hero_b)
policy_net_b = modele_hero(n_observations_b+2, n_actions).to(device)
target_net_b = modele_hero(n_observations_b+2, n_actions).to(device)
if model_name_reload != '' and not only_atq:
    policy_net_b.load_state_dict(torch.load('./models/' + model_name_reload + '_dfs', map_location=device))
    target_net_b.load_state_dict(torch.load('./models/' + model_name_reload + '_dfs', map_location=device))
else:
    target_net_b.load_state_dict(policy_net_b.state_dict())
optimizer_b = optim.AdamW(policy_net_b.parameters(), lr=LR_b, amsgrad=True)
memory_dfs = ReplayMemory(33000)

if is_adversaire:
    adversaire = Joueur('./models/' + model_name_adversaire, n_observations_a, n_observations_b+2, n_actions-1, device, modele_adv)


steps_done = 0
score_attaque = []
score_defense = []
num_episodes = 600

for i_episode in range(num_episodes):

    # Initialize the environment and get its state
    observation_hero, observation_adversaire, info = env.reset()
    state_hero_a, state_hero_b = observation_hero
    state_adversaire_a, state_adversaire_b = observation_adversaire

    state_hero_a = torch.tensor(state_hero_a, dtype=torch.float32, device=device).unsqueeze(0)
    state_hero_b = torch.tensor(state_hero_b, dtype=torch.float32, device=device).unsqueeze(0)
    state_adversaire_a = torch.tensor(state_adversaire_a, dtype=torch.float32, device=device).unsqueeze(0)
    state_adversaire_b = torch.tensor(state_adversaire_b, dtype=torch.float32, device=device).unsqueeze(0)


    action_a = select_action(state_hero_a, policy_net_a, env.action_space.sample_attaquant, steps_done, device, EPS_END=EPS_END_a, EPS_START=EPS_START_a, EPS_DECAY=EPS_DECAY_a)
    ang1, pow1, _a, _b =  env.get_action((action_a, 0))
    if pow1 == 'SHIELD':
        pow1 = 0
    elif pow1 == 'BOOST':
        pow1 = 650
    state_hero_b = torch.cat([state_hero_b] + [torch.tensor([[entree]]).to(device) for entree in (ang1, pow1)], dim=1)
    
    t=-1
    while True :
        t+=1

        

        action_b = select_action(state_hero_b, policy_net_b, env.action_space.sample_defenseur, steps_done, device, EPS_END=EPS_END_b, EPS_START=EPS_START_b, EPS_DECAY=EPS_DECAY_b)

        steps_done += 1

        if is_adversaire:
            action_adversaire_a, action_adversaire_b = adversaire.action(state_adversaire_a, state_adversaire_b, link_ab=True)
        else:
            action_adversaire_a, action_adversaire_b = 0, 0

        observation_hero, observation_adversaire, reward_atq, reward_dfs, terminated, _ = env.step((action_a.item(), action_b.item()), (action_adversaire_a, action_adversaire_b))
        
        reward_atq = torch.tensor([reward_atq], device=device)
        reward_dfs = torch.tensor([reward_dfs], device=device)

        if terminated:
            next_state_hero_a = None
            next_state_hero_b = None
            state_adversaire_a = None
            state_adversaire_b = None

            n_cp_hero, n_cp_adversaire = env.get_cp()

            score_attaque.append(n_cp_hero)
            score_defense.append(n_cp_adversaire)

            torch.save(policy_net_a.state_dict(), f'./models/{model_name}_atq')
            torch.save(policy_net_b.state_dict(), f'./models/{model_name}_dfs')

            plot_durations(score_attaque, score_defense)
            # if i_episode % 50 == 49:
            #     env.afficher()
            break

        else:
            nxt_st_a, nxt_st_b = observation_hero
            next_state_hero_a = torch.tensor(nxt_st_a, dtype=torch.float32, device=device).unsqueeze(0)
            next_state_hero_b = torch.tensor(nxt_st_b, dtype=torch.float32, device=device).unsqueeze(0)

            stt_adv_a, stt_adv_b = observation_adversaire
            state_adversaire_a = torch.tensor(stt_adv_a, dtype=torch.float32, device=device).unsqueeze(0)
            state_adversaire_b = torch.tensor(stt_adv_b, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory_atq.push(state_hero_a, action_a, next_state_hero_a, reward_atq)
            state_hero_a = next_state_hero_a

            action_a = select_action(state_hero_a, policy_net_a, env.action_space.sample_attaquant, steps_done, device, EPS_END=EPS_END_a, EPS_START=EPS_START_a, EPS_DECAY=EPS_DECAY_a)
            ang1, pow1, _a, _b =  env.get_action((action_a, 0))
            if pow1 == 'SHIELD':
                pow1 = 0
            elif pow1 == 'BOOST':
                pow1 = 650
            next_state_hero_b = torch.cat([next_state_hero_b] + [torch.tensor([[entree]]).to(device) for entree in (ang1, pow1)], dim=1)
            
            memory_dfs.push(state_hero_b, action_b, next_state_hero_b, reward_dfs)
            state_hero_b = next_state_hero_b

            # Perform one step of the optimization (on the policy network)
            optimize_model(memory_atq, policy_net_a, target_net_a, optimizer_a, device, GAMMA=GAMMA_a, BATCH_SIZE=BATCH_SIZE_a, fine_tuning=True)
            optimize_model(memory_dfs, policy_net_b, target_net_b, optimizer_b, device, GAMMA=GAMMA_b, BATCH_SIZE=BATCH_SIZE_b, fine_tuning=True)



            soft_update(policy_net_a, target_net_a, TAU=TAU_a)
            soft_update(policy_net_b, target_net_b, TAU=TAU_b)


print('Complete')
plot_durations(score_attaque, score_defense, show_result=True)
plt.ioff()
plt.show()