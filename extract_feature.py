import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Read image data
# Red tomato
classes = {'Red': 0, 'Yellow': 1, 'Maroon': 2}
is_training = False
if is_training:
    data_folder = 'training/'
else:
    data_folder = 'testing/'



# plt.figure()
# img = cv2.imread('data/training/Tomato Maroon/2_100.jpg', 0)
# plt.hist(img.ravel(),256,[0,256])
# plt.figure()
# img = cv2.imread('data/training/Tomato Red/2_100.jpg', 0)
# plt.hist(img.ravel(),256,[0,256])
# plt.figure()
# img = cv2.imread('data/training/Tomato Yellow/6_100.jpg', 0)
# plt.hist(img.ravel(),256,[0,256]); plt.show()


def his_extract(img):
    # extract histogram on H channel and S channel
    his_H = cv2.calcHist([img], [0], None, [256], [0, 256])
    his_S = cv2.calcHist([img], [1], None, [256], [0, 256])

    his = np.concatenate((his_H, his_S))

    # Normalize histogram
    his = np.true_divide(his, img.shape[0]*img.shape[1]*2)
    return his


for id_cls, cls in enumerate(list(classes.keys())):
    cls_name = cls + '/'
    list_files = glob.glob('data/' + data_folder + cls_name + '*.jpg')
    data = []
    for id, im_pth in enumerate(list_files):
        # write to label file
        image = cv2.imread(im_pth)
        img_name = im_pth.split('/')[-2] + '_' + im_pth.split('/')[-1].split('.')[0]

        # Apply Gaussian to remove noise
        image_blr = cv2.blur(image, (5, 5))

        # Change to HSV color and extract histogram feature
        image_hsv = cv2.cvtColor(image_blr, cv2.COLOR_BGR2HSV)
        histogram = his_extract(image_hsv)

        # store to file
        feature_pth = 'data/' + data_folder + 'feature/'
        label_pth = 'data/' + data_folder + 'label/'
        if not os.path.exists(feature_pth):
            os.makedirs(feature_pth)
        if not os.path.exists(label_pth):
            os.makedirs(label_pth)

        store_pth = feature_pth + img_name + '.npy'
        np.save(store_pth, histogram)
        with open(label_pth + 'label.txt', 'a') as lf:
            if (id_cls >= len(list(classes.keys())) - 1) and (id >= len(list_files)):
                lf.writelines(store_pth + ' ' + str(classes[im_pth.split('/')[-2]]))
            else:
                lf.writelines(store_pth + ' ' + str(classes[im_pth.split('/')[-2]]) + '\n')
            lf.close()
