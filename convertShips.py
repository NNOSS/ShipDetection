from PIL import Image
import loadCSV
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

FILEPATH = '/Data/ShipDetection/'
CSV_FILENAME = FILEPATH + 'train_ship_segmentations.csv'
DATA_FILEPATH = FILEPATH + 'train/'
TRAIN_LABEL_SAVE = FILEPATH + 'train_labels_6'
TRAIN_INPUT_SAVE = FILEPATH + 'train_images_6'
DIVIDEND = 4
WIDTH = 768
HEIGHT = 768
NEW_HEIGHT = HEIGHT / DIVIDEND
NEW_WIDTH = WIDTH / DIVIDEND
batch_size= 2000

def append_binary_file(file_name, bytes_):
    with open(file_name,"ab") as f:
        f.write(bytes_)

files_dict_gen = loadCSV.load_csv(CSV_FILENAME,batch_size)
file_dict = next(files_dict_gen, None)
count = 0
while file_dict:
    # input_group_div = np.zeros((len(file_dict)*DIVIDEND*DIVIDEND,NEW_WIDTH,NEW_HEIGHT,3),dtype=np.uint8)
    # label_group_div = np.full((len(file_dict)*DIVIDEND*DIVIDEND,NEW_WIDTH,NEW_HEIGHT),.1,dtype=np.float32)
    input_group_div = np.zeros((len(file_dict),WIDTH,HEIGHT,3),dtype=np.uint8)
    label_group_div = np.full((len(file_dict),DIVIDEND,DIVIDEND),.1,dtype=np.float32)
    for i, temp in enumerate(file_dict.items()):
        k,v = temp
        try:
            image = Image.open(DATA_FILEPATH + k)
            np_image = np.array(image,dtype=np.uint8)
            for j in range(DIVIDEND*DIVIDEND):
                x_a = (j // DIVIDEND) * NEW_WIDTH
                y_a = (j % DIVIDEND) * NEW_HEIGHT
                label_group[i*DIVIDEND*DIVIDEND + j] = v[x_a:x_a + NEW_WIDTH, y_a:y_a + NEW_HEIGHT]
                input_group[i*DIVIDEND*DIVIDEND + j] = np_image[x_a:x_a + NEW_WIDTH, y_a:y_a + NEW_HEIGHT]
            count+= 1
            if not count%1000:
                print('Count: ' + str(count))
        except:
            continue

    # print(input_group)
    # print(label_group)
    append_binary_file(TRAIN_INPUT_SAVE,input_group.tobytes())
    append_binary_file(TRAIN_LABEL_SAVE,label_group.tobytes())
    file_dict = next(files_dict_gen, None)
print('DOOOOOOOOONE')
