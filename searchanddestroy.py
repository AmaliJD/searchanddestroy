import numpy as np
import random

'''
0 = flat
1 = hilly
2 = forest
3 = caves
'''

def generateBoard(dim, board):
    
    for i in range(dim):
        for j in range(dim):
            board[i,j] = random.randint(0, 3)
            
    return board

dimension = 10
board = np.zeros((dimension, dimension))
board = generateBoard(dimension, board)

print(board)
