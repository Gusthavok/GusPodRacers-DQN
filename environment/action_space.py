from numpy.random import randint

vitesses = [0, 50, 100] # rajouter 'BOOST', 'SHIELD'
angles = [18, 9, 0, -9, -18]
n = len(vitesses) * len(angles) + 1

def sample_defenseur():
    # mime un comportement pas stupide de défenseur

    return randint(0, n)

def sample_attaquant():
    return randint(0, n)