from PIL import Image
import loadCSV
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

FILEPATH = '/Data/ShipDetection/'
CSV_FILENAME = FILEPATH + 'train_ship_segmentations.csv'
DATA_FILEPATH = FILEPATH + 'train/'
TRAIN_LABEL_SAVE = FILEPATH + 'train_labels'
TRAIN_INPUT_SAVE = FILEPATH + 'train_images'
TRAIN_INPUT_SAVE_COARSE = FILEPATH + 'train_images_coarse'
DIVIDEND = 12
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
    # input_group = np.zeros((len(file_dict)*DIVIDEND*DIVIDEND,NEW_WIDTH,NEW_HEIGHT,3),dtype=np.uint8)
    # label_group = np.full((len(file_dict)*DIVIDEND*DIVIDEND,NEW_WIDTH,NEW_HEIGHT),.1,dtype=np.float32)
    input_group = np.zeros((len(file_dict),WIDTH,HEIGHT,3),dtype=np.uint8)
    coarse_label = np.zeros((len(file_dict),DIVIDEND,DIVIDEND),dtype=np.uint8)
    fine_label = np.zeros((len(file_dict),WIDTH,HEIGHT),dtype=np.uint8)

    for i, temp in enumerate(file_dict.items()):
        k,v = temp
        # try:
        image = Image.open(DATA_FILEPATH + k)
        np_image = np.array(image,dtype=np.uint8)
        # input_group[i] = np_image
        # fine_label[i] = v
        for j in range(DIVIDEND*DIVIDEND):
            # x_a = (j // DIVIDEND) * NEW_WIDTH
            # y_a = (j % DIVIDEND) * NEW_HEIGHT
            # label_group[i*DIVIDEND*DIVIDEND + j] = v[x_a:x_a + NEW_WIDTH, y_a:y_a + NEW_HEIGHT]
            # input_group[i*DIVIDEND*DIVIDEND + j] = np_image[x_a:x_a + NEW_WIDTH, y_a:y_a + NEW_HEIGHT]
            x_a = (j // DIVIDEND)
            y_a = (j % DIVIDEND)
            has_boat = np.amax(v[x_a * NEW_WIDTH:x_a * NEW_WIDTH + NEW_WIDTH, y_a * NEW_HEIGHT:y_a * NEW_HEIGHT + NEW_HEIGHT])
            if has_boat and x_a > 0 and y_a > 0:
                im = Image.fromarray(np_image[(x_a-1) * NEW_WIDTH:(x_a+1) * NEW_WIDTH + NEW_WIDTH, (y_a-1) * NEW_HEIGHT:(y_a+1) * NEW_HEIGHT + NEW_HEIGHT])
                im.show()
                raw_input(" ")

            coarse_label[i, x_a, y_a] = has_boat
            # if has_boat:
            #     print('HAS BOAT: ' + str(has_boat))
            #     print('X: ' + str(x_a))
            #     print('Y: ' + str(y_a))
            #     image.show()
            #     im = Image.fromarray(coarse_label[i]* 254)
            #     im = im.resize((300,300))
            #     im.show()
            #     raw_input(" ")


        count+= 1
        if not count%100:
            print('Count: ' + str(count))
        # except:
        #     continue

    # print(input_group)
    # print(label_group)
    # append_binary_file(TRAIN_INPUT_SAVE,input_group.tobytes())
    # append_binary_file(TRAIN_LABEL_SAVE,fine_label.tobytes())
    exit()
    append_binary_file(TRAIN_INPUT_SAVE_COARSE,coarse_label.tobytes())
    file_dict = next(files_dict_gen, None)
print('DOOOOOOOOONE')
