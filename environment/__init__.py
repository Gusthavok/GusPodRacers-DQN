import environment.game.main as game
import environment.action_space as action_space
import environment.ressources.maps as maps

def reset(choose_map = -1):
    global jeu, carte_cp, score_adv
    # Charge une nouvelle partie aléatoire
    # choisir une carte aleatoirement :
    
    if choose_map < 0:
        carte_cp, score_adv = maps.get_random_map()
    else: 
        carte_cp = maps.echantillon_carte[choose_map]
        score_adv = 1

    jeu = game.Etat_de_jeu(carte_cp, nb_tour=100)
    observation_hero, observation_adversaire = (jeu.get_observation(indice = 0), jeu.get_observation(indice = 1)), (jeu.get_observation(indice = 2), jeu.get_observation(indice = 3))

    return (observation_hero, observation_adversaire, False)

translate_action = [
    (angle, vitesse) for vitesse in action_space.vitesses for angle in action_space.angles
] + [(0, 'SHIELD')]
def get_action(action):
    indice1, indice2 = action
    angle1, vitesse1 = translate_action[indice1]
    angle2, vitesse2 = translate_action[indice2]
    return angle1, vitesse1, angle2, vitesse2

def step(action_hero, action_adversaire):
    global jeu
    
    observation_hero, observation_adversaire, reward1, reward2, terminated = jeu.etape_de_jeu(get_action(action_hero), get_action(action_adversaire))
    n_cp = jeu.get_n_cp()

    return observation_hero, observation_adversaire, reward1, reward2, terminated, False

def get_cp():
    global carte_cp, score_adv
    n_cp = len(carte_cp)
    n_cp_hero = jeu.get_n_cp(joueur=0)
    n_cp_adversaire = jeu.get_n_cp(joueur=2)
    print(n_cp_hero, n_cp_adversaire)
    return n_cp_hero/score_adv, n_cp_adversaire/score_adv


def afficher():
    global jeu

    jeu.afficher()