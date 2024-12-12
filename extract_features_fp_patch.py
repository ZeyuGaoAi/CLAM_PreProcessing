import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np
from glob import glob

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Patch_Bag_FP
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_path, loader, model, verbose = 0):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))

	# mode = 'w'
	features_list = []
	indexs_list = []
	inst_labels_list = []
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord']
			inst_labels = data['inst_label'].tolist()
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			features_list.append(features)
			indexs_list += coords
			inst_labels_list += inst_labels

	features_list = np.concatenate(features_list)
	asset_dict = {'feature': features_list, 'index': indexs_list, 'inst_label':inst_labels_list}
	np.save(output_path, asset_dict)

		# asset_dict = {'features': features, 'coords': coords}
		# save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
		# mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--patch_ext', type=str, default= '.jpg')
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'dsmil_lung', 'dsmil_camel', 'uni_v1', 'conch_v1'])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--suffix', type=str, default="_1_512")
parser.add_argument('--patch_size', type=int, default=2048)
args = parser.parse_args()


if __name__ == '__main__':
	print('initializing dataset')

	bags_dataset = glob(os.path.join(args.data_dir, '*'))
	
	os.makedirs(args.feat_dir, exist_ok=True)

	dest_files = os.listdir(args.feat_dir)

	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
			
	_ = model.eval()
	model = model.to(device)
	total = len(bags_dataset)

	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

	for bag_candidate_idx in tqdm(range(total)):
		slide_id = bags_dataset[bag_candidate_idx].split('/')[-1]
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		# if not args.no_auto_skip and slide_id+'.pt' in dest_files:
		if not args.no_auto_skip and slide_id + args.suffix + '.npy' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		# output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		output_path = os.path.join(args.feat_dir, slide_id + args.suffix + '.npy')
		time_start = time.time()
		dataset = Patch_Bag_FP(file_dir=os.path.join(bags_dataset[bag_candidate_idx], '*/*' + args.patch_ext),
									 ori_patch_size=args.patch_size,
									 img_transforms=img_transforms)
		
		# print(dataset[0])
		# continue

		loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)

		
		output_file_path = compute_w_loader(output_path, loader = loader, model = model, verbose = 1)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

		record = np.load(output_file_path, allow_pickle=True)
		print('features size: ', record[()]['feature'].shape)
		print('coordinates size: ', len(record[()]['index']))

		# with h5py.File(output_file_path, "r") as file:
		# 	features = file['features'][:]
		# 	print('features size: ', features.shape)
		# 	print('coordinates size: ', file['coords'].shape)

		# features = torch.from_numpy(features)
		# bag_base, _ = os.path.splitext(bag_name)
		# torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))



