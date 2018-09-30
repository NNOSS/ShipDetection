import numpy as np

def find_islands(array):
    x_len = len(array)
    y_len = len(array[0])
    indices = np.nonzero(array)
    visited = set()
    islands = []
    queue = []
    for x,y in zip(*indices):
        if (x,y) in visited or array[x][y] ==0:
            continue
        else:
            visited.add((x,y))
            queue.append((x,y))
            islands.append([[x,y]])
        while len(queue):
            x, y = queue.pop(0)#yes I know this is O(n)
            for i in range(9):
                x_p = x + i//3 - 1
                y_p = y + i%3 - 1
                if (0 <= x_p < x_len) and (0 <= y_p < y_len):
                    if (x_p,y_p) in visited:
                        continue
                    elif array[x_p][y_p]:
                        visited.add((x_p,y_p))
                        queue.append((x_p,y_p))
                        islands[-1].append([x_p,y_p])
                    else:
                        visited.add((x_p,y_p))
    return islands

def get_boxes(array):
    ship_list = find_islands(array)
    box_list = []
    # print(ship_list)
    for ship in ship_list:
        mins = np.min(ship, axis = 0)
        # print(mins)
        maxes = np.max(ship, axis = 0)
        box_list.append((mins, maxes))
    return box_list

if __name__ == '__main__':
    print(find_islands([\
        [0,1,0,0,0],
        [0,0,0,0,1],
        [0,0,0,0,0],
        [0,1,1,0,0],
        [0,1,1,0,0]
        ]))

    print(find_islands([\
        [0,1,0,1,0,0,0,0,0,0],
        [0,1,0,1,0,0,0,0,0,0],
        [0,1,0,1,0,0,0,0,0,0],
        [0,1,0,1,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,1,1,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,1,0,1,0],
        [0,0,0,0,0,1,1,0,0,1],
        ]))
