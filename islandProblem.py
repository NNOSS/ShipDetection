import numpy as np

def find_islands(array):
    x_len = len(array)
    y_len = len(array[0])
    indices = np.nonzero(array)
    visited = {}
    islands = []
    for x,y in zip(*indices):
        print(x,y)





if __name__ == '__main__':
    find_islands( [[0,1,0],[1,0,1]])
