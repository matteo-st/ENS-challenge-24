import os
import pickle
import random
from random import choice

from pathlib import Path
from sklearn.model_selection import KFold

import cv2
import numpy as np
import pandas as pd
import torch
from batchgenerators.transforms.abstract_transforms import (Compose,
                                                            RndTransform)
from batchgenerators.transforms.crop_and_pad_transforms import \
    RandomCropTransform
from batchgenerators.transforms.spatial_transforms import (MirrorTransform,
                                                           SpatialTransform)
from batchgenerators.utilities.file_and_folder_operations import *
from torch.utils.data.dataset import Dataset

from .utils import *


def gather_keypoints(image):
    # normalize the image
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    sift = cv2.SIFT_create()
    # image = np.squeeze(image, 0) # remove the batch dimension
    keypoints, features = sift.detectAndCompute(image, None)
    # convert keypoints to numpy array
    # the order is y, x
    keypoints = np.array([[kp.pt[1], kp.pt[0]] for kp in keypoints])
    keypoints = np.round(keypoints).astype(np.int32)

    if keypoints.shape[0] == 0:
        return None, None

    # random permute the keypoints and features
    perm = np.random.permutation(len(keypoints))
    keypoints = keypoints[perm]
    features = features[perm]
    return keypoints, features


def manhattan_distance(keypoints1, keypoints2):
    # Expand the keypoints matrices to have a third axis
    keypoints1 = keypoints1[:, np.newaxis, :]
    keypoints2 = keypoints2[np.newaxis, :, :]
    # Compute the difference in x and y coordinates for all pairs of keypoints
    dx = np.abs(keypoints2[:, :, 0] - keypoints1[:, :, 0])
    dy = np.abs(keypoints2[:, :, 1] - keypoints1[:, :, 1])
    # Compute the Manhattan distance for all pairs of keypoints
    distances = dx + dy
    return distances


def find_match(keypoints1, feature1, keypoints2, feature2):
    # find the match between two images
    # keypoints1: keypoints of image 1
    # feature1: feature of image 1
    # keypoints3: keypoints of image 3
    # feature3: feature of image 3
    # return the match keypoints of image 1 and image 3

    # compute manhattan distance between two keypoints
    distance = manhattan_distance(keypoints1, keypoints2)

    # compute the feature distance between two images
    feature_distance = np.linalg.norm(
        feature1[:, np.newaxis, :] - feature2[np.newaxis, :, :], axis=2
    )

    # compute the match
    weighted_distance = (distance < 20) * feature_distance + \
        (distance >= 20) * np.max(feature_distance)
    match1 = np.argmin(weighted_distance, axis=1)
    for i in range(len(keypoints1)):
        neighbor = distance[i] < 20
        if np.sum(neighbor) == 0:
            match1[i] = -1

    match2 = np.argmin(weighted_distance, axis=0)
    for i in range(len(keypoints2)):
        neighbor = distance[:, i] < 20
        if np.sum(neighbor) == 0:
            match2[i] = -1
    # convert to the format used in SuperGlue
    match_list_1 = []
    match_list_2 = []
    missing_list_1 = []
    missing_list_2 = []
    for i, match2_index in enumerate(match1):
        if match2_index == -1:
            missing_list_1.append(i)
        elif match2[match2_index] == i:
            match_list_1.append(i)
            match_list_2.append(match2_index)
        else:
            missing_list_1.append(i)
    for i, match1_index in enumerate(match2):
        if match1_index == -1:
            missing_list_2.append(i)
        elif match1[match1_index] == i:
            # match_list_2.append(i)
            pass
        else:
            missing_list_2.append(i)

    return match1, match2, match_list_1, match_list_2, missing_list_1, missing_list_2


def keypoints_cv2(keypoints):
    # convert the keypoint to cv2 keypoint
    cv2_keypoints = [
        cv2.KeyPoint(x=float(pt[1]), y=float(pt[0]), size=1) for pt in keypoints
    ]
    return cv2_keypoints


def visualize_keypoints(image, keypoints, save_path):
    cv2_keypoints = [
        cv2.KeyPoint(x=float(pt[1]), y=float(pt[0]), size=1) for pt in keypoints
    ]

    cv2_image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    cv2_image = cv2_image.astype(np.uint8)

    image_keypoints = cv2.drawKeypoints(np.squeeze(cv2_image, 0),
                                        cv2_keypoints, None,
                                        color=(0, 255, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(save_path, image_keypoints)


def visualize_array(array, save_path):
    array = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255
    array = array.astype(np.uint8)
    cv2.imwrite(save_path, array)


def label_to_keypoints(label, num_keypoints):
    keypoint_mask = []
    class_coordinates = []

    # Get the coordinates for each class using a list comprehension
    for class_label in range(num_keypoints):
        class_position = (np.argwhere(label[0] == class_label))
        if class_position.shape[0] == 0:
            keypoint_mask.append(0)
        elif class_position.shape[0] == 1:
            class_coordinates.append(class_position)
            keypoint_mask.append(1)
        elif class_position.shape[0] > 1:
            # Get the mean of the coordinates
            keypoint = np.mean(class_position, axis=0, keepdims=True,
                               dtype=np.int32)
            class_coordinates.append(keypoint)
            keypoint_mask.append(1)
        else:
            raise ValueError("Something went wrong")

    # Concatenate the coordinates into a single NumPy array
    coordinates_array = np.vstack(class_coordinates)
    keypoint_mask = np.array(keypoint_mask)
    return coordinates_array, keypoint_mask


def make_array_image(array):
    array = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255
    array = array.astype(np.uint8)
    return array


def visualize_corespondence(image1, image2, keypoints1, keypoints2, save_path):
    matches_np = np.stack(
        [
            np.arange(keypoints1.shape[0]),
            np.arange(keypoints1.shape[0]),
            np.ones(keypoints1.shape[0])
        ],
        axis=1
    )
    matches_cv2 = [
        cv2.DMatch(int(queryIdx), int(trainIdx), 0, float(distance)) for queryIdx, trainIdx, distance in matches_np
    ]
    matched_image = cv2.drawMatches(
        image1[0], keypoints_cv2(keypoints1), image2[0], keypoints_cv2(keypoints2), matches_cv2, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(save_path, matched_image)


def visulize_matches(image1, keypoints1, match1, image2, keypoints2, match2, save_path):

    matches_list = []
    for i, match in enumerate(match1):
        if match != -1:
            matches_list.append([i, match, 1])
    matches_np = np.array(matches_list)
    matches_cv2 = [
        cv2.DMatch(int(queryIdx), int(trainIdx), 0, float(distance)) for queryIdx, trainIdx, distance in matches_np
    ]
    matched_image = cv2.drawMatches(
        image1, keypoints_cv2(keypoints1), image2, keypoints_cv2(keypoints2), matches_cv2, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(save_path, matched_image)


class RaidiumUnlabeled(Dataset):

    def __init__(self, path, purpose=None, args=None):
        
        if args == None:
            self.path = path
            self.image_files = np.array([f for f in os.listdir(path) if f.endswith('.png')])
        else: 
            self.data_dir = args.data_dir
            self.patch_size = args.patch_size
            self.purpose = purpose
            self.classes = args.classes


    def _sup_process(self, image, label, debug=False):
        keypoints, _ = gather_keypoints(image)

        if keypoints is None:
            return None, None, None

        # convert keypoints to label array
        label_array = np.ones_like(image) * (-1)
        label_array[keypoints[:, 0], keypoints[:, 1]] = np.arange(
            keypoints.shape[0]
        )
        if debug:
            print("debug visualizze")
            visualize_keypoints(image[None], keypoints, 'keypoints.png')
        # resize image
        image, coord = pad_and_or_crop(image, self.patch_size, mode='random')
        label_array, _ = pad_and_or_crop(
            label_array, self.patch_size, mode='fixed', coords=coord
        )
        label, _ = pad_and_or_crop(
            label, self.patch_size, mode='fixed', coords=coord
        )
        # convert the label back to keypoints
        keypoints1, keypoints1_mask = label_to_keypoints(
            label_array[None], num_keypoints=keypoints.shape[0]
        )
        keypoints1_mask = (keypoints1_mask == 1)
    
        return image, label, keypoints1

    def __getitem__(self, index):
        debug = False
        img = np.array(cv2.imread(
            os.path.join(self.path,  self.image_files[index]),
            cv2.IMREAD_GRAYSCALE), dtype=np.float32)
        
        # img, label, keypoints = self._sup_process(img, label)

        img = (img - np.min(img)) / (np.max(img) - np.min(img)) 
        return {"slice" : img[None], # add the chanel dimension 
                "image_file" : self.image_files[index]
                }
     
    def __len__(self):
        return len(self.image_files)

class RaidiumLabeled(Dataset):

    def __init__(self, files, purpose, args):
        self.data_dir = args.data_dir
        self.labels = pd.read_csv(
                os.path.join(self.data_dir, "y_train.csv"), 
                index_col=0, 
                ).T
        self.patch_size = args.patch_size
        self.purpose = purpose
        self.classes = args.classes
        self.do_contrast = args.do_contrast
        self.files = files
        
        print(f'dataset length: {len(self.files)}')


    def _sup_process(self, image, label, debug=False):
        keypoints, _ = gather_keypoints(image)

        if keypoints is None:
            return None, None, None

        # convert keypoints to label array
        label_array = np.ones_like(image) * (-1)
        label_array[keypoints[:, 0], keypoints[:, 1]] = np.arange(
            keypoints.shape[0]
        )
        if debug:
            print("debug visualizze")
            visualize_keypoints(image[None], keypoints, 'keypoints.png')
        # resize image
        image, coord = pad_and_or_crop(image, self.patch_size, mode='random')
        label_array, _ = pad_and_or_crop(
            label_array, self.patch_size, mode='fixed', coords=coord
        )
        label, _ = pad_and_or_crop(
            label, self.patch_size, mode='fixed', coords=coord
        )
        # convert the label back to keypoints
        keypoints1, keypoints1_mask = label_to_keypoints(
            label_array[None], num_keypoints=keypoints.shape[0]
        )
        keypoints1_mask = (keypoints1_mask == 1)
    
        return image, label, keypoints1

    def __getitem__(self, index):
        debug = False
        img = np.array(cv2.imread(
            os.path.join(self.data_dir, "x_train", self.files[index]),
            cv2.IMREAD_GRAYSCALE), dtype=np.float32)
        
        label = self.labels.loc[self.files[index]].values.reshape((512,512))
        img, label, keypoints = self._sup_process(img, label)

        img = (img - np.min(img)) / (np.max(img) - np.min(img)) 
        img = img[None] # add the chanel dimension
        return img, label, keypoints
    
    def __len__(self):
        return len(self.files)


def get_split_raidium(data_dir, fold, seed=12345): 
    # image_files = list(sorted(Path(data_dir).glob("*.png"), key=lambda filename: int(filename.name.rstrip(".png"))))
    image_files = np.array([f for f in os.listdir(os.path.join(data_dir, "x_train")) if f.endswith('.png')])
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    splits = kf.split(image_files)
    for i, (train_idx, test_idx) in enumerate(splits):
        train_files = image_files[train_idx]
        test_files = image_files[test_idx]
        if i == fold:
            break
    return train_files, test_files

def raidius_sup_collate(batch):
    """_summary_
    
    Args:
        batch (_type_): _description_

    Returns:
        _type_: _description_
    """
    # if None in batch:
    #     print('None in batch')
    #     batch = [item for item in batch if item is not None]

    # num_keypoints = min([len(item[2]) for item in batch])
    # batch = [list(item) for item in batch]
    # for i in range(len(batch)):
    #     selected_rows = np.random.choice(
    #         batch[i][2].shape[0], size=num_keypoints, replace=False
    #     )
    #     batch[i][2] = batch[i][2][selected_rows]
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/afs/crc.nd.edu/user/d/dzeng2/data/chd/preprocessed_without_label/")
    parser.add_argument("--patch_size", type=tuple, default=(512, 512))
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--do_contrast", default=True, action='store_true')
    parser.add_argument("--slice_threshold", type=float, default=0.05)
    args = parser.parse_args()

    train_keys = os.listdir(os.path.join(args.data_dir, 'train'))
    train_keys.sort()
    train_dataset = CHD(keys=train_keys, purpose='train', args=args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=30,
                                                   shuffle=True,
                                                   num_workers=8,
                                                   drop_last=False)

    pp = []
    n = 0
    for batch_idx, tup in enumerate(train_dataloader):
        print(f'the {n}th minibatch...')
        img1, img2, slice_position, partition = tup
        batch_size = img1.shape[0]
        # print(f'batch_size:{batch_size}, slice_position:{slice_position}')
        slice_position = slice_position.contiguous().view(-1, 1)
        mask = (torch.abs(slice_position.T.repeat(batch_size, 1) -
                slice_position.repeat(1, batch_size)) < args.slice_threshold).float()
        # count how many positive pair in each batch
        for i in range(mask.shape[0]):
            pp.append(mask[i].sum()-1)
        n = n + 1
        if n > 100:
            break
    pp = np.asarray(pp)
    pp_mean = np.mean(pp)
    pp_std = np.std(pp)
    print(f'mean:{pp_mean}, std:{pp_std}')
