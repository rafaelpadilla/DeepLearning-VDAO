https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
import os
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier


directory = '/home/rafael/thesis/VDAO/vdao_alignment_object/shortest_distance/conv1/feats_fold_1_conv1'
neg_path = os.path.join(directory,'neg')
negative_npys_paths = glob.glob(neg_path+'/*.npy')

pos_path = os.path.join(directory,'pos')
positive_npys_paths = glob.glob(pos_path+'/*.npy')

negative_npys_paths_100 = negative_npys_paths[0:100]
positive_npys_paths_100 = positive_npys_paths[0:100]

X = []
for f in negative_npys_paths_100:
    X.append(np.load(f).flatten())
for f in positive_npys_paths_100:
    X.append(np.load(f).flatten())

teste_pos = []
teste_neg = []
for f in positive_npys_paths[100:200]:
    teste_pos.append(np.load(f).flatten())
for f in negative_npys_paths[100:200]:
    teste_neg.append(np.load(f).flatten())

Y = [0]*len(negative_npys_paths_100)+[1]*len(positive_npys_paths_100)
rnd_clf = RandomForestClassifier()
rnd_clf.fit(X,Y)

print(rnd_clf.feature_importances_)

