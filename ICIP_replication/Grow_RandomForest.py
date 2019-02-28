import os
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

target_objects = ['shoe', 'towel', 'brown box', 'black coat', 'black backpack', 'dark-blue box', 'camera box', 'white jar', 'pink bottle']
folds = {'fold_1': target_objects[0],
                'fold_2': target_objects[1],
                'fold_3': target_objects[2],
                'fold_4': target_objects[3],
                'fold_5': target_objects[4],
                'fold_6': target_objects[5],
                'fold_7': target_objects[6],
                'fold_8': target_objects[7],
                'fold_9': target_objects[8]
}

layers = ['conv1','residual1','residual2','residual3','residual4','residual5',
          'residual6','residual7','residual8','residual9','residual10','residual11',
          'residual12','residual13','residual14','residual15','residual16']

X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)

import random

random.seed(0)
np.random.seed(0)

# Loop through every fold
for fold in folds:
    # Loop through every layer
    for layer in layers:
        directory_features = f'/media/storage/VDAO/vdao_alignment_object/shortest_distance/features/{fold}/{layer}'
        # Get paths of negative samples
        neg_path = os.path.join(directory_features,'neg')
        negative_npys_paths = glob.glob(neg_path+'/*.npy')
        # Get paths of positive samples
        pos_path = os.path.join(directory_features,'pos')
        positive_npys_paths = glob.glob(pos_path+'/*.npy')
        # Shuffle features inside their classes
        random.shuffle(negative_npys_paths)
        random.shuffle(positive_npys_paths)

        # Balance by the class with the lowest number of features  
        min_features = min(len(negative_npys_paths), len(positive_npys_paths))
        # Get only the first min_features
        negative_npys_paths = negative_npys_paths[0:10]
        positive_npys_paths = positive_npys_paths[0:10]

        X = []
        for f in negative_npys_paths:
            X.append(np.load(f).flatten())
        for f in positive_npys_paths:
            X.append(np.load(f).flatten())
        
        # Changing X from list to array
        X = np.array(X)
        # Create ground truth labels
        Y = np.array([0]*len(negative_npys_paths)+[1]*len(positive_npys_paths))
        # Apply random forest classification
        rnd_clf = RandomForestClassifier()
        rnd_clf.fit(X,Y)
        # Check accuracy on training data
        y_pred = rnd_clf.predict(X)
        accuracy = accuracy_score(Y, y_pred)
        # Print out results
        print(f'\nTraining: {fold}')
        print(f'Layer: {layer}')
        print(f'Accuracy: {accuracy}')

print('\n Done!')

        
