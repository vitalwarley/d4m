"""
This file contains utility methods that I have used or I am using for deep learning,
more especifically medical imagem segmentation.

Qua, 31 Out 2018.
"""

import os
import zipfile
import numpy as np

from PIL import Image

db_folder = None
raw_train_dir = None
raw_label_dir = None
p1_train_dir = None
p1_label_dir = None

def set_imgs_dirs(root):
    # Change to directory with dataset (this notebook location)

    # Defining actual folder and new folder where only useful images will go
    db_folder = os.path.join(root, "CVC-ColonDB/CVC-ColonDB/CVC-ColonDB/")

    # Raw
    raw_train_dir = os.path.join(root, "raw/train")
    raw_label_dir = os.path.join(root, "raw/train_masks")

    # Processed: cropped
    p1_train_dir = os.path.join(root, "proc_1/train")
    p1_label_dir = os.path.join(root, "proc_1/train_masks")

    dirs = {'other' : [root, db_folder], 'raw': [raw_train_dir, raw_label_dir], 'processed' : [p1_train_dir, p1_label_dir] }

    return dirs

def check_dirs(dirs):
    for k in dirs.keys():
        for d in dirs[k]:
            if not os.path.exists(d):
                os.makedirs(d)
            else:
                print("{} already created.".format(d))


def unzip_colondb(filename):
    if os.path.exists(filename) and not os.path.exists(db_folder):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            unzipped_file = zip_ref.namelist()[0]
            zip_ref.extractall(unzipped_file)
    else:
        print("{} already extracted.".format(filename))

    # Wait?
    assert os.path.exists(db_folder), "{} doesn't exists.".format(db_folder)
    # Originals, mask and contour images + .DStore = 1141 files
    assert len(os.listdir(db_folder)) == 1141, "Not all files have been extracted!"


"""
 Defining frames full names:
     Original images have <id>.tiff
     Mask images have p<id>.tiff
"""
def get_img_names(extension):

    filenames_original = dict()
    filenames_mask = dict()

    # Each interval from i to s defines a sequence of frames
    # For example: sequence 1 goes from 1.tiff to 39.tiff (same for its polyp masks)
    sequences = [1, 2, 3, 5, 6, 7, 9, 10, 11, 14, 15]
    inf_limits = [1, 39, 61, 77, 98, 149, 156, 204, 209, 264, 274]
    sup_limits = [39, 61, 77, 98, 149, 156, 204, 209, 264, 274, 301]
    limits = [(i, s) for (i, s) in zip(inf_limits, sup_limits)]

    # Actually, the images are in .tiff format, but soon we will have to use .jpeg

    for (seq, lim) in zip(sequences, limits):
        filenames_original[seq] = ['.'.join([str(id), extension])
                                   for id in range(lim[0], lim[1])]
        filenames_mask[seq] = ['.'.join(['p' + str(id), extension])
                                   for id in range(lim[0], lim[1])]

    # Checking the corretude of the file names
    assert filenames_original[1][0] == "{}.{}".format(1, extension)
    assert filenames_mask[1][0] == 'p{}.{}'.format(1, extension)

    return filenames_original, filenames_mask


def get_cropped_imgs(orig, mask):
    # Now we begin to set thing up to crop black borders
    gray = orig.convert('L')
    np_mask = (np.array(gray) > 10).nonzero()
    xs, xe = min(np_mask[0]), max(np_mask[0])
    ys, ye = min(np_mask[1]), max(np_mask[1])
    box = (ys, xs, ye, xe)
    # Crop
    ocropped = orig.crop(box)
    mcropped = mask.crop(box)

    return ocropped, mcropped


def get_out_names(original, mask, out_ext):
    # Out names
    oprefix = original.split('.')[0]
    mprefix = mask.split('.')[0]
    original_out = '.'.join([oprefix, out_ext])
    mask_out = '.'.join([mprefix, out_ext])

    return original_out, mask_out

# Only once.
# Convert .tiff to .jpeg, because we don't have a decoder for .tiff in tensorflow
# AND (updated 27/10)
# Crop images
def process_imgs(folder, actual_ext='tiff', out_ext='jpeg', crop=False):
    originals = []
    masks = []
    original_names = []
    mask_names = []

    fn_original, fn_mask = imutils.get_img_names(actual_ext)

    for (seq_o, seq_m) in zip(fn_original.values(), fn_mask.values()):
        for (original, mask) in zip(seq_o, seq_m):
            # Out images
            o_im = Image.open(os.path.join(folder, original))
            m_im = Image.open(os.path.join(folder, mask))

            if crop:
                o_cropped, m_cropped = get_cropped_imgs(o_im, m_im)
                # Append
                originals.append(o_cropped)
                masks.append(m_cropped)
            else:
                originals.append(o_im)
                masks.append(m_im)

            o_out, m_out = get_out_names(original, mask, out_ext)

            original_names.append(o_out)
            mask_names.append(m_out)

    return originals, masks, original_names, mask_names


def save_imgs(imgs, fnames, dst, extension):
    for img, fn in zip(imgs, fnames):
        img.save(os.path.join(dst, fn), extension, quality=100)


def check_number_imgs(folder):
    if len(os.listdir(folder)) == 300:
        print("{} OK!".format(folder))
        return True
    else:
        print("{} not OK!".format(folder))
        return False


def main():
    # We are inside 2_review/deep_learning here.
    dirs = set_imgs_dirs()
    check_dirs(dirs)
    unzip_colondb()  # Only needed once

    if not (check_number_imgs(raw_train_dir) or check_number_imgs(raw_label_dir)):
        # Raw images
        r_origs, r_masks, r_orig_fns, r_mask_fns = process_imgs(db_folder)
        # Save raw
        save_imgs(r_origs, r_orig_fns, raw_train_dir, "JPEG")
        save_imgs(r_masks, r_mask_fns, raw_label_dir, "JPEG")

    if not (check_number_imgs(p1_train_dir) or check_number_imgs(p1_label_dir)):
        # Processed 1: crop
        p1_origs, p1_masks, p1_orig_fns, p1_mask_fns = process_imgs(db_folder, crop=True)
        # Save processed 1
        save_imgs(p1_origs, p1_orig_fns, proc1_train_dir, "JPEG")
        save_imgs(p1_masks, p1_mask_fns, proc1_label_dir, "JPEG")

