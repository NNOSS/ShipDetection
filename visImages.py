import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import csv

FILEPATH = '/Data/ShipDetection/'
DATA_FILEPATH = FILEPATH + 'train/'
CSV_FILENAME = DATA_FILEPATH +'train_ship_segmentations.csv'
WIDTH = 768
HEIGHT = 768


with open(CSV_FILENAME) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print('Column names are' + ", ".join(row))
        elif line_count < 300:
            line_count += 1
            continue
        else:
            print('Image: ' + row[0])
            print('Values: ' + row[1])
            img=mpimg.imread(DATA_FILEPATH + row[0])
            img.setflags(write=1)
            gt_array = row[1].split(' ')
            if len(gt_array) > 1:
                imgplot = plt.imshow(img)
                plt.show()
            for val in range(len(gt_array)//2):
                index = int(gt_array[2*val])-1
                length = int(gt_array[2*val+1])
                x = index % WIDTH
                y = index // WIDTH
                img[x:x+length,y] = np.full((length,3), 255)

            if len(gt_array) > 1:
                imgplot = plt.imshow(img)
                plt.show()
        line_count += 1

    print('Lines Processed: ' + str(line_count))
