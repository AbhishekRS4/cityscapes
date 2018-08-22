# @author : Abhishek R S
# script for generating l-r flip images, labels for training

import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf

tf.enable_eager_execution()

def get_image(img_file):
    img_string = tf.read_file(img_file)
    img = tf.image.decode_png(img_string)
    
    return img

def data_augment(src_images_dir, src_labels_dir):
    tar_images_dir = src_images_dir
    tar_labels_dir = src_labels_dir

    images_list = os.listdir(src_images_dir)
    print('Number of files to process : ' + str(len(images_list)))

    for img_file in images_list:
        img_0 = get_image(os.path.join(src_images_dir, img_file))
        lbl_0 = get_image(os.path.join(src_labels_dir, img_file.replace('leftImg8bit', 'label')))
    
        img_1 = tf.image.flip_left_right(img_0)
        lbl_1 = tf.image.flip_left_right(lbl_0)
        
        cv2.imwrite(os.path.join(tar_images_dir, img_file.split('.')[0] + '_1.png'), cv2.cvtColor(img_1.numpy(), cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(tar_labels_dir, img_file.replace('leftImg8bit', 'label').split('.')[0] + '_1.png'), lbl_1.numpy())

def main():
    src_images_dir = '/opt/data/abhi/cityscapes/resized_images/train/'
    src_labels_dir = '/opt/data/abhi/cityscapes/resized_labels/train/'

    parser = argparse.ArgumentParser()
    parser.add_argument('-src_images_dir', default = src_images_dir, type = str, help = 'path to load source image files')
    parser.add_argument('-src_labels_dir', default = src_labels_dir, type = str, help = 'path to load source label files')
 
    input_args = vars(parser.parse_args(sys.argv[1:]))

    print('Creating l-r flips for images and labels')
    for k in input_args.keys():
        print(k + ': ' + str(input_args[k]))
    print('')
    print('')

    print('Data augmentation started.....')
    data_augment(input_args['src_images_dir'], input_args['src_labels_dir'])
    print('Data augmentation completed')

if __name__ == '__main__':
    main()
