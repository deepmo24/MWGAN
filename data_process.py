import os
from glob import glob
import shutil
import argparse
import numpy as np
from skimage import feature,io
from skimage import img_as_ubyte
from tqdm import tqdm

def divide_data_by_attributes(attr_path, source_dir, target_dir, selected_attrs):
    """
    Divide the CelebA data to different domains according to their attributes.
    """
    # read data
    lines = [line.rstrip() for line in open(attr_path, 'r')]
    all_attr_names = lines[1].split()

    # record attributes
    attr2idx = {}
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i

    # create target directories
    for attr_name in selected_attrs:
        tgt_dir_test = os.path.join(target_dir, 'test', attr_name, 'images')
        tgt_dir_train = os.path.join(target_dir, 'train', attr_name, 'images')
        if not os.path.exists(tgt_dir_test):
            os.makedirs(tgt_dir_test)
        if not os.path.exists(tgt_dir_train):
            os.makedirs(tgt_dir_train)

    # divide data according to selected attributes
    lines = lines[2:]
    for i, line in enumerate(tqdm(lines)):
        split = line.split()
        filename = split[0]
        values = split[1:]
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            if values[idx] == '1':
                src_path = os.path.join(source_dir, filename)
                # test and train
                if (i + 1) < 2000:
                    tgt_path = os.path.join(target_dir, 'test', attr_name, 'images', filename)
                else:
                    tgt_path = os.path.join(target_dir, 'train', attr_name, 'images', filename)
                # copy path
                shutil.copy(src_path, tgt_path)

def extract_edge(source_dir, target_dir, select_nums):
    """
    Transfer natural images to edge images by an edge detection algorithm.
    """
    # read data
    img_list = glob(os.path.join(source_dir, '*'))

    # create target directory
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # extract edges
    max_num = min(len(img_list), select_nums)
    for i in tqdm(range(max_num)):
        img_path = img_list[i]
        img = io.imread(img_path, as_gray=True)
        # extracting method
        edge = feature.canny(img, sigma=0.5)
        height = edge.shape[0]
        width = edge.shape[1]

        image = np.zeros([height,width,3])
        image[:, :, 0] = 1 - edge
        image[:, :, 1] = 1 - edge
        image[:, :, 2] = 1 - edge

        # save edge image
        base_name = os.path.basename(img_path)
        target_path = os.path.join(target_dir, base_name)
        io.imsave(target_path,img_as_ubyte(image))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--process', type=str, default='celeba', choices=['celeba', 'edge'])
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--source_dir', type=str, default='data/celeba/images')
    parser.add_argument('--target_dir', type=str, default='data/Celeba5domain')
    parser.add_argument('--selected_attrs', nargs='+', default=['Black_Hair', 'Blond_Hair', 'Eyeglasses', 'Mustache', 'Pale_Skin'])
    parser.add_argument('--select_nums', type=int, default=10000)
    opts = parser.parse_args()

    if opts.process == 'celeba':
        print('Begin processing...')
        divide_data_by_attributes(opts.attr_path, opts.source_dir, opts.target_dir, opts.selected_attrs)
    else:
        print('Begin processing...')
        for mode in ['train', 'test']:
            for attr_name in opts.selected_attrs:
                source_dir = os.path.join(opts.source_dir, mode, attr_name, 'images')
                target_dir = os.path.join(opts.target_dir, mode, 'Edge', 'images')
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                extract_edge(source_dir, target_dir, opts.select_nums)
