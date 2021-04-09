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


def getFindBoard(board, beliefs):
    find = beliefs.copy()
    dim = len(board[0])
    probNeg = 0

    for i in range(dim):
        for j in range(dim):
            terrain = board[i,j]
            
            if terrain == 0:
                probNeg = .1
            elif terrain == 1:
                probNeg = .3
            elif terrain == 2:
                probNeg = .7
            elif terrain == 3:
                probNeg = .9
            
            find[i,j] = beliefs[i,j] * (1-probNeg)
    
    return find

# agents are given the same start query
def agent1(board, target):
    dim = len(board[0])

    # create initial probablity grid
    beliefs = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            beliefs[i,j] = 1/(dim*dim)
    print("Beliefs: \n",beliefs)

    moves = 1
    distanceTravelled = 0

    # get initial query at random
    query = [random.randint(0, dim-1), random.randint(0, dim-1)]
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
    dim = len(board[0])

    # create initial probablity grid
    beliefs = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            beliefs[i,j] = 1/(dim*dim)
    print("Beliefs: \n",beliefs)

    moves = 1
    distanceTravelled = 0

    # get initial query at random
    query = [random.randint(0, dim-1), random.randint(0, dim-1)]
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
    dim = len(board[0])

    # create initial probablity grid
    beliefs = np.zeros((dim, dim))
    # create a distance array to track distance(Manhattan) from current cell to all other cells
    distances = np.zeros((dim, dim))
    # create array of optimal cells to search
    optimal = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            beliefs[i, j] = 1 / (dim * dim)
    print("Beliefs: \n", beliefs)

    moves = 1
    distanceTravelled = 0

    # get initial query at random
    query = [random.randint(0, dim - 1), random.randint(0, dim - 1)]
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
        probNeg = 0
        if terrain == 0:
            probNeg = .1
        elif terrain == 1:
            probNeg = .3
        elif terrain == 2:
            probNeg = .7
        elif terrain == 3:
            probNeg = .9

        # update probability on found cell
        print("Prob: ", prob, "Terrain: ", terrain)
        beliefs[query[0], query[1]] = probNeg * beliefs[query[0], query[1]]

        # sum all probs
        totalBelief = 0
        for i in range(dim):
            for j in range(dim):
                totalBelief += beliefs[i, j]
        # update belief table
        for i in range(dim):
            for j in range(dim):
                beliefs[i, j] /= totalBelief
        print("New Beliefs:\n", beliefs)

        # update cell distances
        for i in range(dim):
            for j in range(dim):
                if not [i, j] == query:
                    distances[i][j] = abs(query[0] - i) + abs(query[1] - j)
                else:
                    distances[i][j] = 0
        print("New distances:\n", distances)

        # update optimal matrix to determine which cell to search next based on belief and distance
        for i in range(dim):
            for j in range(dim):
                optimal[i][j] = beliefs[i][j] / distances[i][j]
        print("Optimal cells:\n", optimal)

        # find largest value in optimal, and search that cell next
        largest = 0
        nextSearch = [-1][-1]
        for i in range(dim):
            for j in range(dim):
                if optimal[i][j] > largest:
                    largest = optimal
                    nextSearch = [i][j]
        print("The best cell to search next is :\n", nextSearch)
        print("Distance to the next cell is:\n", distances[nextSearch[0]][nextSearch[1]])
        moves+=1
        distanceTravelled+=distances[nextSearch[0]][nextSearch[1]]
        query = nextSearch
        
    return 0
    

dimension = 4
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
