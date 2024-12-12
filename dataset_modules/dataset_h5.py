import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import h5py
import os
import cv2
import openslide
from glob import glob

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			roi_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.roi_transforms = img_transforms
		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		with h5py.File(self.file_path, "r") as hdf5_file:
			dset = hdf5_file['imgs']
			for name, value in dset.attrs.items():
				print(name, value)

		print('transformations:', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		anno_path,
		slide_file_path,
		ori_patch_size,
		img_transforms=None,
		fp_rate=0.1):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.slide_file_path = slide_file_path
		self.wsi = openslide.open_slide(slide_file_path)
		self.roi_transforms = img_transforms
		self.fp_rate = fp_rate
		self.ori_patch_size = ori_patch_size

		self.file_path = file_path
		self.anno_path = anno_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
		
		self.summary()

		h,w = self.wsi.level_dimensions[0]

		if self.anno_path is not None:
			cancer_mask = cv2.imread(anno_path)
			cancer_mask = cv2.cvtColor(cancer_mask, cv2.COLOR_BGR2RGB)
			cancer_mask_binary = np.zeros(cancer_mask.shape[:-1])
			cancer_mask_binary[(cancer_mask!=[0,0,0]).any(axis=-1)] = 1
			self.cancer_mask_binary=cancer_mask_binary.T
			self.mask_h, self.mask_w = self.cancer_mask_binary.shape
			self.mag = int(h/self.mask_h)
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):

		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]

		try:
			img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		except Exception as e:
			print(f"Error reading region at {coord}: {e}. Reloading slide.")
			self.wsi = openslide.open_slide(self.slide_file_path)
			img = Image.new('RGB', (self.patch_size, self.patch_size), color='white')

		x, y = coord

		if self.anno_path is not None:
			x_mag = x//self.mag
			y_mag = y//self.mag
			size_mag = self.patch_size//self.mag
			mask = self.cancer_mask_binary[x_mag:x_mag+size_mag,y_mag:y_mag+size_mag]

			counts = np.count_nonzero(mask)
			if counts / mask.size >= self.fp_rate:
				inst_label = 1
			else:
				inst_label = 0
		else:
			inst_label = -1
				
		img = self.roi_transforms(img)
		# img = self.roi_transforms(images=img, return_tensors="pt")

		return {'img': img, 'coord': '{}_{}_{}.png'.format(x, y, self.ori_patch_size), 'inst_label': inst_label}
	
class Patch_Bag_FP(Dataset):
	def __init__(self,
		file_dir,
		ori_patch_size,
		img_transforms=None,
		fp_rate=0.1):
		"""
		Args:
			file_dir (string): Path to the .jpg file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.roi_transforms = img_transforms
		self.fp_rate = fp_rate
		self.ori_patch_size = ori_patch_size

		self.file_dir = file_dir

		self.img_path_list = glob(file_dir)

		self.length = len(self.img_path_list)
			
	def __len__(self):
		return self.length

	def __getitem__(self, idx):

		img_path = self.img_path_list[idx]

		basename = os.path.basename(img_path)

		img = Image.open(img_path).convert('RGB')
		# this is for sicapv2
		# x, y = basename.split('_')[-3], basename.split('_')[-2].split('.')[0]

		inst_label = int(img_path.split('/')[-2])
				
		img = self.roi_transforms(img)

		# '{}_{}_{}.png'.format(x, y, self.ori_patch_size)

		return {'img': img, 'coord': basename, 'inst_label': inst_label}


class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




