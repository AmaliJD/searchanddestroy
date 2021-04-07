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

# get false negative probabilty for a terrain
def getFalseNegative(terrain):
    falseNeg = 0
    if terrain == 0:
        falseNeg = .1
    elif terrain == 1:
        falseNeg = .3
    elif terrain == 2:
        falseNeg = .7
    elif terrain == 3:
        falseNeg = .9

    return falseNeg
    
# calculate probability of getting a fail at a cell
def getProbabilityOfFail(p, terrain):
    falseNeg = getFalseNegative(terrain)
    
    probability_fail = (p*falseNeg) + (1-p)
    return probability_fail
    

# calculate P(in J | fail J)
def ProbJ_FailJ(p, terrain):
    falseNeg = getFalseNegative(terrain)

    denominator = getProbabilityOfFail(p, terrain)
    probability = (p*falseNeg) / denominator
    return probability

# calculate P(in I | fail J)
def ProbI_FailJ(pi, pj, terraini, terrainj):
    falseNeg = getFalseNegative(terraini)

    denominator = getProbabilityOfFail(pj, terrainj)
    probability = pi / denominator
    return probability

# calculate P(Observations)
def UpdateBoardSimulation(cell, simulatedBeliefs, board):
    dim = len(simulatedBeliefs[0])
    
    # update rest of cells probabillity -> create new simulatedBeliefs
    for i in range(dim):
        for j in range(dim):
            if not [i,j] == cell:
                simulatedBeliefs[i, j] = ProbI_FailJ(simulatedBeliefs[i, j], simulatedBeliefs[cell[0], cell[1]], board[i, j], board[cell[0], cell[1]])

    simulatedBeliefs[cell[0], cell[1]] = ProbJ_FailJ(simulatedBeliefs[cell[0], cell[1]], board[cell[0], cell[1]])

    print("Simulated Beliefs:\n", simulatedBeliefs)
    return simulatedBeliefs


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


def getFindBoard(board, beliefs):
    find = beliefs.copy()
    dim = len(board[0])

    for i in range(dim):
        for j in range(dim):
            terrain = board[i,j]
            
            falseNeg = getFalseNegative(terrain)
            
            find[i,j] = beliefs[i,j] * (1-falseNeg)
    
    return find


def agent1(board, target):
    print("\n-------- AGENT 1 --------\n")
    dim = len(board[0])

    # create initial probablity grid
    beliefs = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            beliefs[i,j] = 1/(dim*dim)
    print("Beliefs: \n",beliefs)

    Observations = []
    moves = 1
    distanceTravelled = 0

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
        Observations.append([query[0], query[1]])
        
        # update probability on found cell
        print("Prob: ", prob, "Terrain: ", terrain, "\nObservations: ", Observations)

        # update all cells probabillity
        simulatedBeliefs = beliefs.copy()
        for cell in Observations:
            simulatedBeliefs = UpdateBoardSimulation(cell, simulatedBeliefs, board)
                
        beliefs = simulatedBeliefs
        print("New Beliefs:\n", beliefs)

        ''' get new query '''
        # query from neighbor with highest probability of having target
        maxProbability = 0
        cell = query
        ''' immediate neighbors version
        for cell in getNeighbors(beliefs, query):
            if beliefs[cell[0], cell[1]] > maxProbability:
                maxProbability = beliefs[cell[0], cell[1]]
                query = cell
        '''
        for i in range(dim):
            for j in range(dim):
                if beliefs[i, j] > maxProbability:
                    maxProbability = beliefs[i, j]
                    cell = [i, j]
                elif beliefs[i, j] == maxProbability:
                    dist1 = abs(query[0] - cell[0]) + abs(query[1] - cell[1])
                    dist2 = abs(query[0] - i) + abs(query[1] - j)

                    if dist2 < dist1:
                        cell = [i, j]
                    elif dist2 == dist1 and random.randint(0, 1) == 0:
                        cell = [i, j]

        dist = abs(query[0] - cell[0]) + abs(query[1] - cell[1])
        query = cell
        print("\nQuery: ", query)
        print("Distance: ", dist)

        # get probability at cell & terrain type
        prob = beliefs[query[0], query[1]]
        terrain = (int)(board[query[0], query[1]])
        moves += 1
        distanceTravelled += dist
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
    
    return moves, distanceTravelled


'''
Agent 2 is exactly like agent 1 except it's criteria for choosing a new cell to query
'''
def agent2(board, target):
    print("\n-------- AGENT 2 --------\n")
    dim = len(board[0])

    # create initial probablity grid
    beliefs = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            beliefs[i,j] = 1/(dim*dim)
    print("Beliefs: \n",beliefs)

    Observations = []
    moves = 1
    distanceTravelled = 0

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
        Observations.append([query[0], query[1]])
        
        # update probability on found cell
        print("Prob: ", prob, "Terrain: ", terrain, "\nObservations: ", Observations)
        
        # update all cells probabillity
        simulatedBeliefs = beliefs.copy()
        for cell in Observations:
            simulatedBeliefs = UpdateBoardSimulation(cell, simulatedBeliefs, board)
                
        beliefs = simulatedBeliefs
        print("New Beliefs:\n", beliefs)



        ''' get new query '''
        # generate find probability board
        findBoard = getFindBoard(board, beliefs)
        print("Find Prob:\n", findBoard)
        
        # query from neighbor with highest probability of FINDING target
        maxProbability = 0
        cell = query
        '''
        for cell in getNeighbors(findBoard, query):
            if findBoard[cell[0], cell[1]] > maxProbability:
                maxProbability = findBoard[cell[0], cell[1]]
                query = cell
        '''
        for i in range(dim):
            for j in range(dim):
                if findBoard[i, j] > maxProbability:
                    maxProbability = findBoard[i, j]
                    cell = [i, j]
                elif findBoard[i, j] == maxProbability:
                    dist1 = abs(query[0] - cell[0]) + abs(query[1] - cell[1])
                    dist2 = abs(query[0] - i) + abs(query[1] - j)

                    if dist2 < dist1:
                        cell = [i, j]
                    elif dist2 == dist1 and random.randint(0, 1) == 0:
                        cell = [i, j]
                
        dist = abs(query[0] - cell[0]) + abs(query[1] - cell[1])
        query = cell
        print("\nQuery: ", query)
        print("Distance: ", dist)

        # get probability at cell & terrain type
        prob = beliefs[query[0], query[1]]
        terrain = (int)(board[query[0], query[1]])
        moves += 1
        distanceTravelled += dist
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
    
    return moves, distanceTravelled


def agent3(board, target):

    '''
    Both agents are kinda shit
    hopefully this one doesn't disappoint
    '''

    return 0
    

dimension = 2
board = np.zeros((dimension, dimension))
board = generateBoard(dimension, board)
target = [random.randint(0, dimension-1), random.randint(0, dimension-1)]

print(board)
print("Target: ", target, "\n")

a1moves, a1dist = agent1(board, target)
print("\n\n")
a2moves, a2dist = agent2(board, target)

print("\nagent1 Moves: ", a1moves, "\nagent2 Moves: ", a2moves)
print("\nagent1 Distance: ", a1dist, "\nagent2 Distance: ", a2dist)
print("\nagent1 Final Score: ", a1moves+a1dist, "\nagent2 Final Score: ", a2moves+a2dist)
