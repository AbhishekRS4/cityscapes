# @author : Abhishek R S
# script for converting mask to label for training

import os
import sys
import numpy as np
import cv2
import argparse

color_maps = [(128, 64, 128), # road
              (244, 35, 232), # sidewalk
              (250, 170, 30), # traffic light
              (220, 220,  0), # traffic sign
              (220, 20, 60), # person
              (255,  0,  0), # rider
              (0,  0, 142), # car
              (0,  0, 70), # truck
              (0, 60, 100), # bus 
              (0,  0, 90), # caravan
              (0,  0, 110),  # trailer
              (0, 80, 100), # train
              (0,  0, 230), # motorcycle
              (119, 11, 32)] # bicycle

# other classes can also be included

def convert_mask_2_label(src_masks_dir, tar_labels_dir):
    src_mask_files = os.listdir(src_masks_dir)
    print('Number of masks to be processed : ' + str(len(src_mask_files)))

    if not os.path.exists(tar_labels_dir):
        os.makedirs(tar_labels_dir)

    for mask_file in src_mask_files:
        mask = cv2.imread(os.path.join(src_masks_dir, mask_file))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        label = np.zeros((mask.shape[0], mask.shape[1]))

        for i in range(1, len(color_maps) + 1):
            label += i * np.all((mask[:, :] == color_maps[i - 1]), axis = 2)

        cv2.imwrite(os.path.join(tar_labels_dir, mask_file.replace('gtFine_color', 'label')), label)

def main():
    src_masks_dir = './resized_masks/valid/'
    tar_labels_dir = './resized_labels/valid/'

    parser = argparse.ArgumentParser()
    parser.add_argument('-src_masks_dir', default = src_masks_dir, type = str, help = 'path to load source mask files')
    parser.add_argument('-tar_labels_dir', default = tar_labels_dir, type = str, help = 'path to save target label files')

    input_args = vars(parser.parse_args(sys.argv[1:])) 
    
    print('Converting source masks to target labels')
    for k in input_args.keys():
        print(k + ': ' + str(input_args[k]))
    print('')
    print('') 

    print('Mask to label conversion started.....')
    convert_mask_2_label(input_args['src_masks_dir'], input_args['tar_labels_dir'])
    print('Mask to label conversion completed')    

if __name__ == '__main__':
    main()
