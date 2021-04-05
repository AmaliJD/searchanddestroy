import numpy as np
import random

'''
0 = flat
1 = hilly
2 = forest
3 = caves

P(fail | T in cell)
0.1 flat
0.3 hilly
0.7 forest
0.9 caves
'''

def generateBoard(dim, board):
    
    for i in range(dim):
        for j in range(dim):
            board[i,j] = random.randint(0, 3)
            
    return board


# calculate new probality given curr probability and terrain type
def getProbabilityOnFail(p, terrain):
    probNeg = 0
    if terrain == 0:
        probNeg = .1
    elif terrain == 1:
        probNeg = .3
    elif terrain == 2:
        probNeg = .7
    elif terrain == 3:
        probNeg = .9

    P = (p*probNeg) / ((p*probNeg) + (1-p))
    return P


def agent1(board, target):
    dim = len(board[0])

    # create initial probablity grid
    beliefs = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            beliefs[i,j] = 1/(dim*dim)
    print("\n",beliefs)

    moves = 0
    query = [random.randint(0, dim-1),random.randint(0, dim-1)]
    
    while query != target:
        moves += 1
        
        prob = beliefs[query[0], query[1]]
        terrain = board[query[0], query[1]]

        # update probability on found cell
        beliefs[query[0], query[1]] = getProbabilityOnFail(prob, terrain)

        # update probabliities on rest of cells
        # idk how to do this yet
        
        # query = new query
        # not like this tho
        query = [random.randint(0, dim-1),random.randint(0, dim-1)]
    
    return moves
    

dimension = 4
board = np.zeros((dimension, dimension))
board = generateBoard(dimension, board)
target = [random.randint(0, dimension-1), random.randint(0, dimension-1)]

print(board)
print(agent1(board, target))
