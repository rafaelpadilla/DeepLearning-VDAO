import os
import glob
import random
import numpy as np
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

random.seed(0)
np.random.seed(0)

BASE_DIR = '/nfs/proc/rafael.padilla'
features_dir = { 'train': os.path.join(BASE_DIR,'vdao_alignment_object/shortest_distance/features/'),
                 'test': os.path.join(BASE_DIR,'vdao_alignment_research/shortest_distance/features/') }
JSON_FILE_RESEARCH = 'vdao_research.json'

def get_objects_info(classes, json_file):
    ret = {}
    with open(json_file, "r") as read_file:
        data = json.load(read_file)
        for t in data['tables']:
            for o in data['tables'][t]['objects']:
                if o['object_class']  in classes:
                    if t not in ret:
                        ret[t] = []
                    ret[t].append(o)
    return ret

def compare_predictions(y_1, y_2):
    # one could also use the code below to obtain the accuracy
    # accuracy = accuracy_score(Y_hat, Y_pred)
    assert len(y_1) == len(y_2)
    incorrect = []
    same = []
    for i in range(len(y_1)):
        if y_1[i] == y_2[i]:
            same.append(i)
        else:
            incorrect.append(i)
    return {'correct_ids': same, 
            'total_correct': len(same),
            'accuracy': len(same)/len(y_1),
            'incorrect_ids': incorrect,
            'total_incorrect': len(incorrect),
            'inaccuracy': len(incorrect)/len(y_1)
            }

def get_predict_probabilities(predict_proba, list_name_features, classes=[0,1]):
    assert len(predict_proba) == len(list_name_features)
    ret = {}
    for id in range(len(predict_proba)):
        ret[list_name_features[id]] = {str(classe): predict_proba[id][classe] for classe in classes}
    return ret

def get_features(_features_dir, max_features_per_class = None, shuffle=True, balance_classes=True):
    # Get paths of negative samples
    neg_path = os.path.join(_features_dir,'neg')
    negative_npys_paths = glob.glob(neg_path+'/*.npy')
    # Get paths of positive samples
    pos_path = os.path.join(_features_dir,'pos')
    positive_npys_paths = glob.glob(pos_path+'/*.npy')
    # Shuffle features inside their classes
    if shuffle:
        random.shuffle(negative_npys_paths)
        random.shuffle(positive_npys_paths)
    # Balance by the class with the lowest number of features
    if balance_classes:
        min_features = min(len(negative_npys_paths), len(positive_npys_paths))
        # Get only the first min_features
        if max_features_per_class == None:
            negative_npys_paths = negative_npys_paths[0:min_features]
            positive_npys_paths = positive_npys_paths[0:min_features]
        else:
            negative_npys_paths = negative_npys_paths[0:max_features_per_class]
            positive_npys_paths = positive_npys_paths[0:max_features_per_class]
    # Obtain features
    X = []
    paths_features = []
    for f in negative_npys_paths:
        X.append(np.load(f).flatten())
        paths_features.append(os.path.split(f)[1])
    for f in positive_npys_paths:
        X.append(np.load(f).flatten())
        paths_features.append(os.path.split(f)[1])
    # Changing X from list to array
    X = np.array(X)
    # Create labels
    Y = np.array([0]*len(negative_npys_paths)+[1]*len(positive_npys_paths))
    return X, Y, paths_features

target_objects = ['shoe', 'towel', 'brown box', 'black coat', 'black backpack', 'dark blue box', 'camera box', 'white jar', 'pink bottle']
folds = {'fold_1': target_objects[0],
         #'fold_2': target_objects[1],
         #'fold_3': target_objects[2],
         #'fold_4': target_objects[3],
         #'fold_5': target_objects[4],
         #'fold_6': target_objects[5],
         #'fold_7': target_objects[6],
         #'fold_8': target_objects[7],
         #'fold_9': target_objects[8]
}

n_estimators = 10

layers = ['conv1','residual1','residual2','residual3','residual4','residual5',
          'residual6','residual7','residual8','residual9','residual10','residual11',
          'residual12','residual13','residual14','residual15','residual16']

# Loop through every fold
for fold in folds:
    # Get target object
    tar_obj = folds[fold]
    # Obtain target objects info into the testing structure
    test_object_info = get_objects_info([tar_obj], JSON_FILE_RESEARCH)
    # Calculate TP and FP for the whole layer considering all tables (videos)
    all_TP = 0
    all_FP = 0
    all_DIS = 0
    # Total number of positive and negative ground truth frames
    total_number_pos_gt_frames = 0
    total_number_neg_gt_frames = 0
    # Loop through every layer
    for layer in layers:
        # Print out status for new training
        print('='*40)
        print('\n')
        print('TRAINING')
        print(f'Fold: {fold}')
        print(f'Layer: {layer}')
        # Define path with features for training
        dir_features_train = os.path.join(features_dir['train'], f'{fold}',f'{layer}')
        # Get features for training
        X, Y_hat, paths_feat_training = get_features(dir_features_train, max_features_per_class=None, shuffle=True, balance_classes=True)
        # Get amount of positives and negatives
        amount_pos = (Y_hat == 1).sum()
        amount_neg = (Y_hat == 0).sum()
        # Apply random forest classification
        rnd_clf = RandomForestClassifier(n_estimators=n_estimators)
        rnd_clf.fit(X,Y_hat)
        # Predict the training data (validation)
        Y_pred = rnd_clf.predict(X)
        # Dictionary to save results of this layer for each one of the 59 tables
        results = {}
        # Obtain accuracy of the training data
        result_training = compare_predictions(Y_hat, Y_pred)
        accuracy = result_training['accuracy']
        # Obtain the probability of each class per prediction
        predict_proba_training = rnd_clf.predict_proba(X)
        # Organize in a dictionary the probabilities, correct/incorrect features
        result_training['predict_proba'] = get_predict_probabilities(predict_proba_training, paths_feat_training, classes=[0,1])
        result_training['correct_predict_features'] = [paths_feat_training[id] for id in result_training['correct_ids']]
        result_training['incorrect_predict_features'] = [paths_feat_training[id] for id in result_training['incorrect_ids']]
        # Add it in the results dictionary
        results['result_training'] = result_training
        # Print out accuracy for training
        print('Dataset distribution (pos, neg): ({}, {}) : ({:.1%}, {:.1%})'.format(amount_pos,amount_neg,amount_pos/len(Y_hat),amount_neg/len(Y_hat)))
        print(f'Training accuracy: {accuracy}')
        print('-'*40)
        print(f'TESTING target [{tar_obj}]:')
        # Test on the resarch database
        overall_Y_test_hat = []
        overall_Y_test_pred = []
        # Dictionary to save the results for testing
        results['result_testing'] = {}
        # First get all feature samples from each table
        for table in test_object_info:
            print('-'*40)
            print(f'Table: [{table}][{layer}][{tar_obj}]')
            # Get directory with features considering the table and layer
            dir_features_test = os.path.join(features_dir['test'],table,layer)
            if not os.path.isdir(dir_features_test):
                print(f'Testing folder {dir_features_test} not found.')
                continue
            # Get features for testing
            X_test, Y_test_hat, paths_feat_testing = get_features(dir_features_test, max_features_per_class=None, shuffle=False, balance_classes=False)
            # Get amount of positives and negatives
            amount_pos = (Y_test_hat == 1).sum()
            amount_neg = (Y_test_hat == 0).sum()
            # Update total number of positive and negative frames
            total_number_pos_gt_frames += amount_pos
            total_number_neg_gt_frames += amount_neg
            # Predict
            Y_test_pred = rnd_clf.predict(X_test)
            # Append prediction and gt for an overall accuracy  measurement
            overall_Y_test_hat += Y_test_hat.tolist()
            overall_Y_test_pred += Y_test_pred.tolist()
            accuracy = accuracy_score(Y_test_hat, Y_test_pred)
            print('Dataset distribution (pos, neg): ({}, {}) : ({:.1%}, {:.1%})'.format(amount_pos,amount_neg,amount_pos/len(Y_test_hat),amount_neg/len(Y_test_hat)))
            print('Accuracy: {:.2%}'.format(accuracy))
            # Add the results in the dictionary
            final_results = {}
            TP = 0
            FP = 0
            for idx in range(len(paths_feat_testing)):
                final_results[paths_feat_testing[idx]] = {}
                final_results[paths_feat_testing[idx]]['groundtruth_class'] = int('pos' in paths_feat_testing[idx])
                assert int(Y_test_hat[idx]) == int('pos' in paths_feat_testing[idx]) # double checking
                final_results[paths_feat_testing[idx]]['predicted_class'] = int(Y_test_pred[idx])
                assert int(Y_test_pred[idx]) == int(rnd_clf.predict(X_test[idx].reshape(1,-1))) # double checking
                final_results[paths_feat_testing[idx]]['predicted_correcly'] = final_results[paths_feat_testing[idx]]['groundtruth_class'] == final_results[paths_feat_testing[idx]]['predicted_class']
                # If an object was predicted in the scene
                if final_results[paths_feat_testing[idx]]['predicted_class']  == 1:
                    # Predicted an object and there was an object
                    if final_results[paths_feat_testing[idx]]['groundtruth_class'] == 1:
                        TP += 1 
                    # Predicted an object but there was no object
                    else:
                        FP += 1
            print('(TP, FP): (%d, %d)' % (TP, FP))
            # TP_rate: TP / (TP+FN)
            # TP is the number of true positives, FP is the number of false negatives and TP+FN is the total number of positives 
            if amount_pos != 0:
                TP_rate = TP / amount_pos            
            else:
                TP_rate = 0
            # FP_rate: FP / (FP+TN)
            # FP is the number of false positives, TN is the number of true negatives and FP+TN is the total number of negatives
            if amount_neg != 0:
                FP_rate = FP / amount_neg
            else:
                FP_rate = 0
            print('(TP rate, FP rate): (%.2f, %.2f)' % (TP_rate, FP_rate))
            DIS = np.sqrt((1-TP_rate)**2 + FP_rate**2)
            print('DIS: %.2f' % DIS)
            # Update FP and TP for the currenct layer
            all_TP += TP
            all_FP += FP
            results_table = {}
            results_table[table] = compare_predictions(Y_test_hat, Y_test_pred)
            results_table[table]['correct_predict_features'] = [paths_feat_testing[id] for id in results_table[table]['correct_ids']]
            results_table[table]['incorrect_predict_features'] = [paths_feat_testing[id] for id in results_table[table]['incorrect_ids']]
            results_table[table]['predict_proba'] = get_predict_probabilities(rnd_clf.predict_proba(X_test), paths_feat_testing, classes=[0,1])
            results_table[table]['summary_results'] = final_results
            results_table[table]['TP'] = TP
            results_table[table]['FP'] = FP
            results_table[table]['TPR'] = TP_rate
            results_table[table]['FPR'] = FP_rate
            results_table[table]['DIS'] = DIS
            # Add table results to the result_testing dict
            results['result_testing'][table] = results_table[table]
        # Obtain the overall accuracy considering features of all tables
        overall_accuracy = accuracy_score(overall_Y_test_hat, overall_Y_test_pred)
        # overall TP and FP rates
        overall_TP_rate = all_TP/total_number_pos_gt_frames
        overall_FP_rate = all_FP/total_number_neg_gt_frames
        all_DIS = np.sqrt((1-overall_TP_rate)**2 + overall_FP_rate**2)
        results['result_testing']['overall_accuracy'] = overall_accuracy
        results['result_testing']['overall_TP'] = all_TP
        results['result_testing']['overall_FP'] = all_FP
        results['result_testing']['overall_TPR'] = overall_TP_rate
        results['result_testing']['overall_FPR'] = overall_FP_rate
        results['result_testing']['overall_DIS'] = all_DIS
        print('-'*40)
        print('Overall accuracy: {:.2%}'.format(overall_accuracy))
        print('(Overall TP, FP): (%d, %d)' % (all_TP, all_FP))
        print('(Overall TPR, FPR): (%.2f, %.2f)' % (overall_TP_rate, overall_FP_rate))
        print('Overall DIS: %.2f' % all_DIS)
        # Save results
        dir_save_results = f'RF_results/{fold}'
        if not os.path.isdir(dir_save_results):
            os.makedirs(dir_save_results)
        path_save_results = os.path.join(dir_save_results, '%s.pkl'%layer)
        f = open(path_save_results,'wb')
        pickle.dump(results,f)
        f.close()
        print('Saving results in: %s' % path_save_results)

print('Finished!')
