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
    denom = (p*probNeg) + (1-p)
    return P, denom


def getNeighbors(board, cell):
    neighbors = []
    dim = len(board[0])
    
    if (cell[1] < dim - 1):
        neighbors.append([cell[0], cell[1] + 1])
    if (cell[0] < dim - 1):
        neighbors.append([cell[0] + 1, cell[1]])
    if (cell[1] > 0):
        neighbors.append([cell[0], cell[1] - 1])
    if (cell[0] > 0):
        neighbors.append([cell[0] - 1, cell[1]])
            
    return neighbors


def getFindBoard(board):
    return 0


def agent1(board, target):
    dim = len(board[0])

    # create initial probablity grid
    beliefs = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            beliefs[i,j] = 1/(dim*dim)
    print("Beliefs: \n",beliefs)

    moves = 1

    # get initial query at random
    query = [random.randint(0, dim-1),random.randint(0, dim-1)]
    print("\nQuery: ", query)

    # get probability at cell & terrain type
    prob = beliefs[query[0], query[1]]
    terrain = (int)(board[query[0], query[1]])
    found = False

    # find target with probability based on terrain
    if terrain == 0:
        found = random.randint(1, 10) > 1
    elif terrain == 1:
        found = random.randint(1, 10) > 3
    elif terrain == 2:
        found = random.randint(1, 10) > 7
    elif terrain == 3:
        found = random.randint(1, 10) > 9

    targetfound = (query == target) and found

    # loop and continue querying
    while not targetfound:
        
        # update probability on found cell
        print("Prob: ", prob, "Terrain: ", terrain)
        beliefs[query[0], query[1]], denominator = getProbabilityOnFail(prob, terrain)

        # update rest of cells probabillity
        for i in range(dim):
            for j in range(dim):
                if not [i,j] == query:
                    prob_i = beliefs[i, j]
                    beliefs[i, j] = prob_i / denominator

        print("New Beliefs:\n", beliefs)



        ''' get new query '''
        # query from neighbor with highest probability of having target
        maxProbability = 0
        for cell in getNeighbors(beliefs, query):
            if beliefs[cell[0], cell[1]] > maxProbability:
                maxProbability = beliefs[cell[0], cell[1]]
                query = cell
                
        print("\nQuery: ", query)

        # get probability at cell & terrain type
        prob = beliefs[query[0], query[1]]
        terrain = (int)(board[query[0], query[1]])
        moves += 1
        found = False

        # find target with probability based on terrain
        if terrain == 0:
            found = random.randint(1, 10) > 1
        elif terrain == 1:
            found = random.randint(1, 10) > 3
        elif terrain == 2:
            found = random.randint(1, 10) > 7
        elif terrain == 3:
            found = random.randint(1, 10) > 9

        targetfound = (query == target) and found
    
    return moves


'''
Agent 2 is exactly like agent 1 except it's criteria for choosing a new cell to query
Complete the getFindBoard() function to produce a board of probabilies of FINDING the target
'''
def agent2(board, target):
    dim = len(board[0])

    # create initial probablity grid
    beliefs = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            beliefs[i,j] = 1/(dim*dim)
    print("Beliefs: \n",beliefs)

    moves = 1

    # get initial query at random
    query = [random.randint(0, dim-1),random.randint(0, dim-1)]
    print("\nQuery: ", query)

    # get probability at cell & terrain type
    prob = beliefs[query[0], query[1]]
    terrain = (int)(board[query[0], query[1]])
    found = False

    # find target with probability based on terrain
    if terrain == 0:
        found = random.randint(1, 10) > 1
    elif terrain == 1:
        found = random.randint(1, 10) > 3
    elif terrain == 2:
        found = random.randint(1, 10) > 7
    elif terrain == 3:
        found = random.randint(1, 10) > 9

    targetfound = (query == target) and found

    # loop and continue querying
    while not targetfound:
        
        # update probability on found cell
        print("Prob: ", prob, "Terrain: ", terrain)
        beliefs[query[0], query[1]], denominator = getProbabilityOnFail(prob, terrain)

        # update rest of cells probabillity
        for i in range(dim):
            for j in range(dim):
                if not [i,j] == query:
                    prob_i = beliefs[i, j]
                    beliefs[i, j] = prob_i / denominator

        print("New Beliefs:\n", beliefs)



        ''' get new query '''
        # generate find probability board
        findBoard = getFindBoard(board)
        
        # query from neighbor with highest probability of FINDING target
        maxProbability = 0
        for cell in getNeighbors(findBoard, query):
            if findBoard[cell[0], cell[1]] > maxProbability:
                maxProbability = findBoard[cell[0], cell[1]]
                query = cell
                
        print("\nQuery: ", query)

        # get probability at cell & terrain type
        prob = beliefs[query[0], query[1]]
        terrain = (int)(board[query[0], query[1]])
        moves += 1
        found = False

        # find target with probability based on terrain
        if terrain == 0:
            found = random.randint(1, 10) > 1
        elif terrain == 1:
            found = random.randint(1, 10) > 3
        elif terrain == 2:
            found = random.randint(1, 10) > 7
        elif terrain == 3:
            found = random.randint(1, 10) > 9

        targetfound = (query == target) and found
    
    return moves
    

dimension = 4
board = np.zeros((dimension, dimension))
board = generateBoard(dimension, board)
target = [random.randint(0, dimension-1), random.randint(0, dimension-1)]

print(board)
print("Target: ", target, "\n")
print(agent1(board, target))
