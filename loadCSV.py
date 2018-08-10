import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import csv

FILEPATH = '/Data/ShipDetection/'
CSV_FILENAME = FILEPATH + 'train_ship_segmentations.csv'
DATA_FILEPATH = FILEPATH + 'train/'
WIDTH = 768
HEIGHT = 768
batch_size= 1000

def load_csv(CSV_FILENAME, batch_size):
    file_dict = {}
    with open(CSV_FILENAME) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print('Column names are' + ", ".join(row))
            else:
                empty_vals = np.zeros((WIDTH,HEIGHT),dtype=bool)
                # print('Image: ' + row[0])
                # print('Values: ' + row[1])
                gt_array = row[1].split(' ')
                for val in range(len(gt_array)//2):
                    index = int(gt_array[2*val])-1
                    length = int(gt_array[2*val+1])
                    x = index % WIDTH
                    y = index // WIDTH
                    empty_vals[x:x+length,y] = np.full((length), True)
                if row[0] in file_dict:
                    file_dict[row[0]] = np.bitwise_or(file_dict.get(row[0]), empty_vals)
                else:
                    file_dict[row[0]] = empty_vals
                if not line_count % batch_size:
                    yield file_dict
                    file_dict = {}
            line_count += 1
    yield file_dict



if __name__ == '__main__':
    files_dict_gen = load_csv(CSV_FILENAME, 1000)
    print(next(files_dict_gen, None))
