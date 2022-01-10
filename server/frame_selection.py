'''
Modified Date: 2022/01/11
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import os
import numpy as np
import random
import tqdm

dataset_name_list = ['Real_life', 'Bag_of_lies']

if __name__ == '__main__':
    data_path = '../../dataset' # Path to data
    number_frames = 10 # How many frames you would like to use to train
    number_selection = 10 # How many groups you want to choose per folder
    dataset_name = dataset_name_list[0]
    random.seed(412) 
    
    save_path = os.path.join(data_path, dataset_name + '.csv')
    save_list = []

    data_folder_path = os.path.join(data_path, dataset_name)
    data_folder_list = os.listdir(data_folder_path)
    data_folder_list.sort()

    for folder_name in tqdm.tqdm(data_folder_list):
        folder_path = os.path.join(data_folder_path, folder_name)
        folder_name_list = os.listdir(folder_path)
        folder_name_list.sort()

        len_frames = len(folder_name_list)
        frame_interval = len_frames // number_frames
        for i in range(number_selection):
            random_frame = []
            for i in range(0, number_frames, 1):
                random_num = random.randint(i * frame_interval, ((i+1) *frame_interval)-1)
                random_frame.append(random_num)
            folder_name_list = np.array(folder_name_list)
            random_frame = np.array(random_frame, dtype=int)
            
            frame_list = folder_name_list[random_frame]
            frame_list = np.insert(frame_list, 0, folder_name)
            save_list.append(frame_list)
    np.savetxt(save_path, save_list, fmt='%s', delimiter=',')
                