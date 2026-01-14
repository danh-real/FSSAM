import sys
import os
import os.path
import cv2
import numpy as np
import copy

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
import glob
from tqdm import tqdm
from PIL import Image

from .get_weak_anns import transform_anns

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split=0, data_root=None, data_list=None, sub_list=None, filter_intersection=False):
    assert split in [0, 1, 2, 3]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2, 
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32    
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        new_label_class = []

        if filter_intersection:  # filter images containing objects of novel categories during meta-training
            if set(label_class).issubset(set(sub_list)):
                for c in label_class:
                    if c in sub_list:
                        tmp_label = np.zeros_like(label)
                        target_pix = np.where(label == c)
                        tmp_label[target_pix[0], target_pix[1]] = 1
                        if tmp_label.sum() >= 2 * 32 * 32:
                            new_label_class.append(c)
        else:
            for c in label_class:
                if c in sub_list:
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c)
                    tmp_label[target_pix[0], target_pix[1]] = 1
                    if tmp_label.sum() >= 2 * 32 * 32:
                        new_label_class.append(c)

        label_class = new_label_class

        if len(label_class) > 0:
            image_label_list.append(item)
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list[c].append(item)

    print("Checking image&label pair {} list done! ".format(split))
    return image_label_list, sub_class_file_list


class Sarcoma(Dataset):
    def __init__(self, split=3, shot=1, data_root=None, base_data_root=None, data_list=None, data_set=None,
                 use_split_coco=False,
                 transform=None, transform_tri=None, mode='train', ann_type='mask',
                 ft_transform=None, ft_aug_size=None):

        assert mode in ['train', 'val']

        self.mode = mode
        self.split = split
        self.shot = shot
        self.data_root = data_root
        self.base_data_root = base_data_root
        self.ann_type = ann_type

        self.categories = ["Mass", "Edema"]
        self.class_list = list(range(2))
        self.sub_list = list(range(2))
        self.sub_val_list = list(range(2))

        mode = 'train' if self.mode == 'train' else 'val'

        fss_list_root = './lists/{}/fss_list/{}/'.format(data_set, mode)
        fss_data_list_path = fss_list_root + 'data_list_{}.txt'.format(split)
        fss_sub_class_file_list_path = fss_list_root + 'sub_class_file_list_{}.txt'.format(split)

        suffix = "Tr" if mode == "train" else "Ts"
        self.img_path = os.path.join(data_root, f"images{suffix}")
        self.ann_path = os.path.join(data_root, f"labels{suffix}")
        
        # Build metadata for sampling
        self.num = 0
        self.img_metadata_classwise = self.build_img_metadata_classwise()

        self.transform = transform
        self.transform_tri = transform_tri
        self.ft_transform = ft_transform
        self.ft_aug_size = ft_aug_size

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        image_path, support_image_path_list, class_id = self.sample_episode(index)
        label_path = image_path.replace("image", "label")
        support_label_path_list = [p.replace("image", "label") for p in support_image_path_list]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        query_parts = image_path.split("/")
        label_class = query_parts[3].split("_")[-1]
        label[np.where(label == 0)] = 0
        label[np.where(label > 0)] = 1

        support_image_list_ori = []
        support_label_list_ori = []
        support_label_list_ori_mask = []
        subcls_list = [class_id]
        
        for k in range(self.shot):
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k]
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            target_pix = np.where(support_label > 0)
            ignore_pix = np.where(support_label == 0)
            support_label[:, :] = 0
            support_label[target_pix[0], target_pix[1]] = 1

            support_label, support_label_mask = transform_anns(support_label, self.ann_type)  # mask/bbox
            support_label[ignore_pix[0], ignore_pix[1]] = 0
            support_label_mask[ignore_pix[0], ignore_pix[1]] = 0
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError(
                    "Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))
            support_image_list_ori.append(support_image)
            support_label_list_ori.append(support_label)
            support_label_list_ori_mask.append(support_label_mask)
        assert len(support_label_list_ori) == self.shot and len(support_image_list_ori) == self.shot

        raw_image = image.copy()
        raw_label = label.copy()
        support_image_list = [[] for _ in range(self.shot)]
        support_label_list = [[] for _ in range(self.shot)]
        if self.transform is not None:
            image, label = self.transform(image, label)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list_ori[k],
                                                                              support_label_list_ori[k])

        s_xs = support_image_list
        s_ys = support_label_list
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

        # Return
        if self.mode == 'train':
            return image, label, s_x, s_y, subcls_list
        elif self.mode == 'val':
            return image, label, s_x, s_y, subcls_list, raw_label
        
    def sample_episode(self, idx):
        class_id = 0 if idx < len(self.img_metadata_classwise["Mass"]) else 1
        class_sample = self.categories[class_id]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        if idx < len(self.img_metadata_classwise["Mass"]):
            query_name = self.img_metadata_classwise["Mass"][idx]
        else:
            query_name = self.img_metadata_classwise["Edema"][idx - len(self.img_metadata_classwise["Mass"])]
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break
        
        return query_name, support_names, class_id

    
    def build_img_metadata_classwise(self):
        """Build metadata: collect all image paths with valid masks."""
        # For sarcoma, we'll use a simpler approach: all slices with any annotation
        # are valid for all categories (tumor and necrosis are both valid targets)
        img_metadata_classwise = {}
        all_valid_images_mass = []
        all_valid_images_edema = []
        
        # Scan all sample folders
        sample_folders = sorted(glob.glob(os.path.join(self.img_path, '*')))
        
        for sample_folder in sample_folders:
            if not os.path.isdir(sample_folder):
                continue
            
            sample_name = os.path.basename(sample_folder)
            label_folder = os.path.join(self.ann_path, sample_name)
            
            if not os.path.exists(label_folder):
                continue
            
            # Get all PNG files in this sample folder
            img_files = sorted(glob.glob(os.path.join(sample_folder, '*.png')))
            
            for img_path in img_files:
                slice_name = os.path.basename(img_path)
                label_path = os.path.join(label_folder, slice_name)
                
                if not os.path.exists(label_path):
                    continue
                
                # Read mask to see if it has any annotation
                mask = np.array(Image.open(label_path).convert('L'))
                
                # Check if mask has any non-zero values (any annotation)
                if np.any(mask > 0):
                    if sample_name.split("_")[-1] == "Mass":
                        all_valid_images_mass.append(img_path)
                        self.num += 1
                    else:
                        all_valid_images_edema.append(img_path)
                        self.num += 1
        
        # Assign all valid images to each category
        # This allows sampling from any category to work
        
        # for cat in self.categories:
        #     img_metadata_classwise[cat] = all_valid_images.copy()
        
        img_metadata_classwise["Mass"] = all_valid_images_mass.copy()
        img_metadata_classwise["Edema"] = all_valid_images_edema.copy()
        
        print(f"Found {len(all_valid_images_mass) + len(all_valid_images_edema)} valid slices with annotations")
        
        return img_metadata_classwise