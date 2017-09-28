# ALL by CHEN Siyu
# PRIVATE LICENSE
# 2017-2018
import os
from tqdm import tqdm
from scipy.io import loadmat
import scipy.misc as sm
import numpy as np


def force_exist(dirname):
    if dirname == '':
        return True
    if not os.path.exists(os.path.dirname(dirname)):
        force_exist(os.path.dirname(dirname))
        print('creating',dirname)

    if not os.path.exists(dirname):
        os.makedirs(dirname)
        return False
    else:
        return True

def list_library(topdir):
    #for i in glob(topdir):
    #    print(i)
    # get absolute filepath of files for each person
    # acceptable structure is:
    # topdir/
    #   ----/person1/1.jpg
    #   ----/person2/1.jpg
    persons=[]
    persons_files={}
    for path,subdirs,files in tqdm(os.walk(topdir)):
        if path == topdir:
            persons = subdirs
        else:
            persons_files[os.path.basename(path)]=[os.path.join(path,file) for file in files]
    if len(persons)>20:
        print('found',len(persons),'persons')
    else:
        print('found persons',*persons)
    return persons_files

def load_all_mats_by_paths(mat_paths,npz_path='mat.npz'):
    real_images =[]
    for mat_path in tqdm(mat_paths):
        mat = loadmat(mat_path)
        # Left eye (batch_size, height, width)
        real_images.extend(mat['data'][0][0][0][0][0][1])
        # Right eye
        real_images.extend(mat['data'][0][0][1][0][0][1])
    print('load_all_mats_by_paths:re-formating')
    real_data = np.stack(real_images, axis=0)
    print('load_all_mats_by_paths:saving')
    np.savez(npz_path, real=real_data)
    return real_data

def load_from_npz(npz_path):
    npz_obj = np.loadmat(open(npz_path,'rb'))
    keys = npz_obj.keys()
    return keys,npz_obj

def save_to_png(img_batch,save_path='Normalized_pngs'):
    force_exist(save_path)
    batch,height,width = img_batch.shape
    for i in tqdm(range(batch)):
        fname = os.path.join(save_path,'%d.png' % i)
        sm.imsave(fname,img_batch[i,:,:])

def main():
    persons_files = list_library('Normalized')
    all_files = []
    for p in tqdm(persons_files):
        all_files.extend(persons_files[p])
    img_batch = load_all_mats_by_paths(all_files)
    save_to_png(img_batch)
    return img_batch

if __name__ == '__main__':
    main()