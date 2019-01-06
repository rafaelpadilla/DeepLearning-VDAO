import numpy as np
import os
import random as rd
import math
import time
import re
import gc
import h5py
from keras.applications import ResNet50
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, AveragePooling2D
from keras.preprocessing import image
from keras.optimizers import Adamax, Adam
from imagenet_utils import preprocess_input
from keras import backend as K

### MACROS ###

LAYER_NAME = np.array(['res2a_branch2a', 'res2b_branch2a', 'res2c_branch2a',
					   'res3a_branch2a', 'res3b_branch2a', 'res3c_branch2a',
					   'res3d_branch2a', 'res4a_branch2a', 'res4b_branch2a',
					   'res4c_branch2a', 'res4d_branch2a', 'res4e_branch2a',
					   'res4f_branch2a', 'res5a_branch2a', 'res5b_branch2a',
					   'res5c_branch2a', 'avg_pool'])

AVGPOOL_SIZE = np.array([21,28,28,28,21,21,21,21,14,14,14,14,14,14,7,7,7])

VDAO_DATABASE_LIGHTING_ENTRIES = np.array(['NORMAL-Light','EXTRA-Light']) 

VDAO_DATABASE_OBJECT_ENTRIES = np.array(['Black_Backpack',
										 'Black_Coat',
										 'Brown_Box',
										 'Camera_Box',
										 'Dark-Blue_Box',
										 'Pink_Bottle',
										 'Shoe',
										 'Towel',
										 'White_Jar',
										 'Mult_Objs1',
										 'Mult_Objs2',
										 'Mult_Objs3'])

VDAO_DATABASE_OBJECT_POSITION_ENTRIES = np.array(['POS1','POS2','POS3']) 

VDAO_DATABASE_REFERENCE_ENTRIES = np.array(['Pink_Bottle;Towel;Black_Coat;Black_Backpack',
											'Shoe;Dark-blue_Box;Camera_Box',
											'White_Jar;Brown_Box',
											'Mult_Objs1',
											'Mult_Objs2;Mult_Objs3'])
											
VDAO_TEST_ENTRIES = np.array(['cameraBox', 'darkBlueBox_1', 'darkBlueBox_2',
							  'pinkBottle', 'shoe', 'towel', 'whiteJar'])




# changeModel - Troca o modelo da ResNet50 e determina a camada de output
def changeModel():

	global ctr_int
	global LAYER_NAME
	
	model = ResNet50(weights='imagenet', include_top=False)
	mid_layer = Model(input = model.input, output = model.get_layer(LAYER_NAME[ctr_int]).input)
	out_layer = Sequential()
	out_layer.add(mid_layer)
	out_layer.add(AveragePooling2D((AVGPOOL_SIZE[ctr_int],AVGPOOL_SIZE[ctr_int])))
	return out_layer


# setup_prog - Define os objetos a serem escolhidos para serem salvos
def setup_prog():

	global countList

	if (countList == 0):
		obj_vec = np.array([0,1,2,3,4,5,6,7,8,9,10,11])

	return obj_vec


def main_prog():

	global rnd_obj_lst
	global feat_imp_vec
	global pwd
	global nb_frames_offset
	global mv_res_dir
	global countList
	global ctr_int
	global nb_objs_test
	global VDAO_DATABASE_LIGHTING_ENTRIES, VDAO_DATABASE_OBJECT_ENTRIES, VDAO_DATABASE_REFERENCE_ENTRIES, VDAO_DATABASE_OBJECT_POSITION_ENTRIES, FRAMES_SRC
	global LAYER_NAME
	global VDAO_TEST_ENTRIES
	global HDF5_DST
	global heightVar, lengthVar


	obj_v_lst = setup_prog()
	first_frame = True
	nb_frames_test = 300
	frames_per_ref = frames_per_obj = math.floor(nb_frames_test / nb_objs_test / 3)
	if (1 == (frames_per_obj % 2)):
		frames_per_obj += 1
		frames_per_ref += 1
	frames_per_obj_P = int(frames_per_obj/2)
	frames_per_ref = frames_per_obj = int(frames_per_obj)
	nb_data_aug_imgs = 1

	str_objs = ''
	for f in obj_v_lst:
		str_objs += str(int(f))

	
	for ctr_int in range(17):
		
		out_layer = changeModel()

		for obj_nb in obj_v_lst:
			for ill_nb in VDAO_DATABASE_LIGHTING_ENTRIES:
				for pos_nb in VDAO_DATABASE_OBJECT_POSITION_ENTRIES:
					obj_src = ill_nb + '/' + \
					VDAO_DATABASE_OBJECT_ENTRIES[obj_nb] + '_' + pos_nb + '_target'

					ref_src = ill_nb + '/' + \
					VDAO_DATABASE_OBJECT_ENTRIES[obj_nb] + '_' + pos_nb + '_reference'

					temp = os.path.join(FRAMES_SRC,obj_src)
					tempr = os.path.join(FRAMES_SRC,ref_src)

					
					if (os.path.exists(temp)):
						os.chdir(temp)	
						frames_list = os.listdir(temp)
						nb_frames = len(frames_list) - nb_frames_offset

						X_train = np.array([])
						y_train = np.array([])

						counterObjTrain = -1
						counterRefTrain = 0

						if obj_nb in [9,10,11]:
							temp_rd_bg = os.path.join(FRAMES_SRC, VDAO_DATABASE_LIGHTING_ENTRIES[1] + '/Random_Background_target')
							temp_rd_bg_r = os.path.join(FRAMES_SRC, VDAO_DATABASE_LIGHTING_ENTRIES[1] + '/Random_Background_reference')
							frames_list = os.listdir(temp_rd_bg)
							nb_frames = len(frames_list)
							rnd_bg_obj_frames = np.array([])


						with open('object_frames.txt','r') as f:
							content = f.readlines()
							content = [x.strip() for x in content]
							if '' in content:
								content = content[0:-1]
							if (len(content) == 2):
								obj_frames = np.array(list(map(int,re.findall('\d+',content[0]))))
								obj_frames = np.concatenate((obj_frames,np.array(list(map(int,re.findall('\d+',content[1]))))),axis=0)
								obj_frames = np.concatenate((np.arange(obj_frames[0],obj_frames[1]+1),np.arange(obj_frames[2],obj_frames[3]+1)), axis=0)
							else:
								obj_frames = np.array(list(map(int,re.findall('\d+',content[0]))))
								obj_frames = np.arange(obj_frames[0],obj_frames[1]+1)

						even_frame_vec = obj_frames
						if not obj_nb in [9,10,11]:
							for sel_fr in range(0,obj_frames.shape[0]):
								val_fr = False
								while (val_fr != True):
									nb_choice = rd.choice(range(0,nb_frames))
									if not nb_choice in even_frame_vec:
										even_frame_vec = np.concatenate((even_frame_vec,np.array([nb_choice,])),axis=0)
										val_fr = True
						else:
							for sel_fr in range(0,obj_frames.shape[0]):
								val_fr = False
								while (val_fr != True):
									nb_choice = rd.choice(range(0,nb_frames))
									if not nb_choice in rnd_bg_obj_frames:
										rnd_bg_obj_frames = np.concatenate((rnd_bg_obj_frames,np.array([nb_choice,])),axis=0)
										val_fr = True
							rnd_bg_obj_frames = rnd_bg_obj_frames.astype(int)

						for i in even_frame_vec:
							os.chdir(temp)
							print('FOLD_NB: ' + str_objs + '__///__' + LAYER_NAME[ctr_int])
							print(temp + '///TRAIN_BATCH /// ' + ill_nb)
							img_path = 'frame_' + str(i) + '.png'
							img = image.load_img(img_path, target_size=(heightVar, lengthVar))
							x = image.img_to_array(img)
							x = np.expand_dims(x,axis=0)
							x = preprocess_input(x)
							X_train_temp = out_layer.predict(x)
							print ('########### FRAME {0} ###########'.format(i))
							counterObjTrain += 1

							if i in obj_frames:
								y_train = np.concatenate((y_train,np.ones((nb_data_aug_imgs,))),axis=0)
							else:
								y_train = np.concatenate((y_train,np.zeros((nb_data_aug_imgs,))),axis=0)

							os.chdir(tempr)
							print('FOLD_NB: ' + str_objs + '__///__' + LAYER_NAME[ctr_int])
							print(tempr + '///TRAIN_BATCH /// ' + ill_nb)
							img_path = 'frame_' + str(i) + '.png'
							img = image.load_img(img_path, target_size=(heightVar, lengthVar))
							x = image.img_to_array(img)
							x = np.expand_dims(x,axis=0)
							x = preprocess_input(x)
							X_train_temp = X_train_temp - out_layer.predict(x)

							if (X_train.size == 0):
								#X_train = np.zeros(X_train_temp.shape)
								X_train = np.zeros((750,X_train_temp.shape[1],X_train_temp.shape[2],X_train_temp.shape[3]))

							X_train[counterObjTrain] = X_train_temp
							print ('########### FRAME {0} ###########'.format(i))
							counterRefTrain += 1

						if obj_nb in [9,10,11]:
							for i in rnd_bg_obj_frames:
								os.chdir(temp_rd_bg)
								print('FOLD_NB: ' + str_objs + '__///__' + LAYER_NAME[ctr_int])
								print(temp_rd_bg + '///TRAIN_BATCH')
								img_path = 'frame_' + str(i) + '.png'
								img = image.load_img(img_path, target_size=(heightVar, lengthVar))
								x = image.img_to_array(img)
								x = np.expand_dims(x,axis=0)
								x = preprocess_input(x)
								X_train_temp = out_layer.predict(x)
								print ('########### FRAME {0} ###########'.format(i))
								counterObjTrain += 1

								y_train = np.concatenate((y_train,np.zeros((nb_data_aug_imgs,))),axis=0)

								os.chdir(temp_rd_bg_r)
								print('FOLD_NB: ' + str_objs + '__///__' + LAYER_NAME[ctr_int])
								print(temp_rd_bg_r + '///TRAIN_BATCH')
								img_path = 'frame_' + str(i) + '.png'
								img = image.load_img(img_path, target_size=(heightVar, lengthVar))
								x = image.img_to_array(img)
								x = np.expand_dims(x,axis=0)
								x = preprocess_input(x)
								X_train_temp = X_train_temp - out_layer.predict(x)
								X_train[counterObjTrain] = X_train_temp
								print ('########### FRAME {0} ###########'.format(i))
								counterRefTrain += 1

						X_train = X_train[0:counterObjTrain+1]

						os.chdir(HDF5_DST)
						print('SAVING VDAO HDF5 ...')
						
						# e.g.: NORMAL_Light_Black_Backpack_POS1_res2a_branch2a_X_TRAIN_SET
						h5_file = h5py.File('article_train_batch_new.hdf5','a')
						dset = h5_file.create_dataset(ill_nb + '_' + VDAO_DATABASE_OBJECT_ENTRIES[obj_nb] + '_' + pos_nb + '_' + LAYER_NAME[ctr_int] + '_X_TRAIN_SET'
						,data=X_train, compression='gzip', compression_opts=4)
						dset2 = h5_file.create_dataset(ill_nb + '_' + VDAO_DATABASE_OBJECT_ENTRIES[obj_nb] + '_' + pos_nb + '_' + LAYER_NAME[ctr_int] + '_y_TRAIN_SET'
						,data=y_train, compression='gzip', compression_opts=4)
						h5_file.close()

			
		K.clear_session()


############ VARIAVEIS QUE PODEM SER ALTERADAS ######################################

# diretorio fonte dos frames da VDAO
FRAMES_SRC = '/home/bruno.afonso/datasets/Reference_Object_frames_skip_17_full_aligned'

# diretorio de destino para salvar o HDF5
HDF5_DST = '/home/bruno.afonso/datasets/article_HDF5'

# variavel de altura/largura do frame que deseja carregar
heightVar = 360
lengthVar = 640


####################################################################################
		
pwd = os.getcwd()
feat_imp_vec = 0
nb_frames_offset = 1

nb_objs_train = 6
nb_objs_test = 9 - nb_objs_train
nb_train_vecs = 4
countList = 0

while (countList < 1):
	main_prog()
	countList += 1

