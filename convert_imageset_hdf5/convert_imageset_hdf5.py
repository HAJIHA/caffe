import h5py, os
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas
import pandas as pd
import random
import caffe
import cv2


def parse_args():
    """Parse input arguments
    """
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('img_root',
                        help='image root path')
    parser.add_argument('list_file',
                        help='list file')
    parser.add_argument('out_file',
                        help='out file')
    parser.add_argument('mean_file',
                        help='mean file')
    parser.add_argument('--use_label',
                        default='cd1,cd9,cd10',
                        help='use label')
    parser.add_argument('--width',
                        default='540',
                        help='image width')
    parser.add_argument('--height',
                        default='540',
                        help='image height')
    parser.add_argument('--channels',
                        default='3',
                        help='channels')

    parser.add_argument('--resize_width',
                        default='800',
                        help='resize_width')
    parser.add_argument('--resize_height',
                        default='600',
                        help='resize_height')
    parser.add_argument('--augment',
                    default='5',
                    help='augment')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    img_root = args.img_root
    list_file = args.list_file
    out_file = args.out_file
    use_label = args.use_label.split(',')
    mean_file = args.mean_file
    width = int(args.width)
    height = int(args.height)
    channels = int(args.channels)
    resize_width = int(args.resize_width)
    resize_height = int(args.resize_height)
    augment = int(args.augment)

    list_data = pd.read_csv(list_file)
    img_paths = list_data["Path"]

    train_shape = (len(img_paths)*augment, channels, height, width )

    # open a hdf5 file and create earrays
    hdf5_file = h5py.File(out_file, mode='w')
    hdf5_file.create_dataset("img", train_shape, np.float32)
    hdf5_file.create_dataset("labels", (len(img_paths)*augment, len(use_label)), np.float32)
    label_set = list_data[use_label]
 

    for j in range(augment-1):
        label_set = label_set.append(list_data[use_label], ignore_index=True)

    #print list_data[use_label][0]
    print label_set.head()
    hdf5_file["labels"][...] = label_set
    mean = np.zeros((resize_height,resize_width,channels),np.float32)

    # loop over image paths
    for i in range(len(img_paths)): #range(len(img_paths)):
        # print how many images are saved every 1000 images
        if i % 1000 == 0 and i > 1:
            print 'Train data Mean Calc: {}/{}'.format(i, len(img_paths))
        path = img_root + img_paths[i]
        img = cv2.imread(path)
        img = cv2.resize(img, (resize_width,resize_height), interpolation=cv2.INTER_CUBIC)
        imgarray = np.array(img)
        mean += imgarray / float(len(img_paths))

        # img = cv2.imread(path)
        # img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
        # mean += img / float(len(img_paths))
    blob = caffe.io.array_to_blobproto(np.asarray(mean.reshape([1,channels,resize_width,resize_height])))
    binaryproto_file = open(mean_file, 'wb+')
    binaryproto_file.write(blob.SerializeToString())
    binaryproto_file.close()

    
  #  cv2.imshow('Mean debug',mean/float(255))
   # cv2.waitKey()

    # augment
    for j in range(augment):
        # loop over image paths
        for i in range(len(img_paths)):
            # print how many images are saved every 1000 images
            if (j*len(img_paths)+i) % 1000 == 0 and (j*len(img_paths)+i) > 1:
                print 'Train data: {}/{}'.format((j*len(img_paths)+i), len(img_paths))

            path = img_root + img_paths[i]
            img = cv2.imread(path)
            img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)

            img = img - mean
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            x = random.randint(0, resize_width-width)
            y = random.randint(0, resize_height-height)
            crop_img = img[y:y+height, x:x+width]

            randint = random.randint(0,1)
            if randint == 1:
                crop_img = cv2.flip(crop_img,1)    
            randint = random.randint(0,1)
            if randint == 1:
                crop_img = cv2.flip(crop_img,0)    

          #  cv2.imshow('debug',crop_img/float(255))
           # cv2.waitKey()
            # axis orders should change
            crop_img = np.rollaxis(crop_img, 2)
            # save the image and calculate the mean so far
            hdf5_file["img"][(j*len(img_paths)+i), ...] = crop_img[None]
         

    # save the mean and close the hdf5 file
    hdf5_file.close()

if __name__ == '__main__':
    main()
