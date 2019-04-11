import csv
import glob
import json
import os
import pickle
import random
import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

random.seed(0)
np.random.seed(0)

BASE_DIR = '/nfs/proc/rafael.padilla'
BASE_DIR = '/media/storage/VDAO'
features_dir = { 'train': os.path.join(BASE_DIR,'vdao_alignment_object/shortest_distance/features/'),
                 'test': os.path.join(BASE_DIR,'vdao_alignment_research/shortest_distance/features/') }
csv_dir = { 'train': os.path.join(BASE_DIR,'vdao_alignment_object/shortest_distance/intermediate_files/'),
            'test': os.path.join(BASE_DIR,'vdao_alignment_research/shortest_distance/intermediate_files/') }
current_dir = os.path.dirname(os.path.abspath(__file__))
JSON_FILE_RESEARCH = os.path.join(current_dir,'vdao_research.json')
JSON_FILE_OBJECT = os.path.join(current_dir,'vdao_object.json')

############# AQUI INICIO ##############
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

target_objects = ['shoe', 'towel', 'brown box', 'black coat', 'black backpack', 'dark-blue box', 'camera box', 'white jar', 'pink bottle']
folds_number = {'fold_1': target_objects[0],
                'fold_2': target_objects[1],
                'fold_3': target_objects[2],
                'fold_4': target_objects[3],
                'fold_5': target_objects[4],
                'fold_6': target_objects[5],
                'fold_7': target_objects[6],
                'fold_8': target_objects[7],
                'fold_9': target_objects[8]
}
#type = 'tar' e 'ref'
def get_file_filters_train(fold, json_file, layer, include_table_folder=False, types=['tar']):
    # raise Exception('conferir aqui se este método está retonando as informacoes corretamente')
    target_class = folds_number[fold]
    search_terms = []
    items = target_objects.copy()
    items.remove(target_class)
    for item in items:
        obj_info = get_objects_info(item, json_file)
        for table in obj_info:
            folder_prefix = '%s/'%table if include_table_folder else ''
            table_number = table.replace('table_','')
            for obj in obj_info[table]:
                name_obj = obj['name']
                obj_num = name_obj.replace('object ', '')
                if 'tar' in types:
                    st = f'{table}/*/*/feat_*_{layer}_t{table_number}_tar_vid{obj_num}_frame_*.npy'
                if 'ref' in types:
                    st = f'{table}/*/*/feat_*_{layer}_t{table_number}_ref_vid{obj_num}_frame_*.npy'
                search_terms.append(st)
    return search_terms

def get_file_filters_test(fold, json_file, layer, include_table_folder=False, types=['tar'], list_consider_tables=None):
    target_class = folds_number[fold]
    search_terms = []
    obj_info = get_objects_info(target_class, json_file)
    for table in obj_info:
        # If parameter list_consider_tables is given, consider only the tables in this list
        if list_consider_tables != None:
            if table not in list_consider_tables:
                continue
        folder_prefix = '%s/'%table if include_table_folder else ''
        table_number = table.replace('table_','')
        for obj in obj_info[table]:
            # Para os videos teste, o nome do arquivo está sempre no object 01 Ex: 'feat_[obj-video38]_conv1_t38_tar_vid01_frame_55.npy'
            # Pois só existe 1 object por tabela
            obj_num = '01'
            if 'tar' in types:
                st = f'{table}/*/*/feat_*_{layer}_t{table_number}_tar_vid{obj_num}_frame_*.npy'
            if 'ref' in types:
                st = f'{table}/*/*/feat_*_{layer}_t{table_number}_ref_vid{obj_num}_frame_*.npy'
            search_terms.append(st)
    return search_terms
############# AQUI FIM ##############

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

def validate_detections(list_name_features, Y_test_pred, Y_test_hat):
    # Dictionary whose keys are the names of the features and the values represent gt classes and detections
    final_results = {}
    TP = 0
    FP = 0
    # Loop through each testing feature
    for idx in range(len(list_name_features)):
        final_results[list_name_features[idx]] = {}
        final_results[list_name_features[idx]]['groundtruth_class'] = int('pos' in list_name_features[idx])
        assert int(Y_test_hat[idx]) == int('pos' in list_name_features[idx]) # double checking
        final_results[list_name_features[idx]]['predicted_class'] = int(Y_test_pred[idx])
        # NAO FAZ SENTIDO VERIFICAR COM O PROEDICT SENDO QUE O VETOR FOI MUDADO COM O VOTING
        # assert int(Y_test_pred[idx]) == int(rnd_clf.predict(X_test[idx].reshape(1,-1))) # double checking
        final_results[list_name_features[idx]]['predicted_correcly'] = final_results[list_name_features[idx]]['groundtruth_class'] == final_results[list_name_features[idx]]['predicted_class']
        # If an object was detected in the scene
        if final_results[list_name_features[idx]]['predicted_class']  == 1:
            # Predicted an object and there was an object
            if final_results[list_name_features[idx]]['groundtruth_class'] == 1:
                TP += 1
            # Predicted an object but there was no object
            else:
                FP += 1
    return final_results, TP, FP

def apply_temporal_voting(Y_predict, window_size=[2,2]):
    classes_temporally_voted = []
    for idx in range(len(Y_predict)):
        # Define left and right limits
        left_start = max(0,idx-window_size[0])
        right_start = min(len(Y_predict),idx+window_size[1]+1)
        # Obtain partial (from left to right)
        window = list(Y_predict[left_start:right_start])
        # Are there more positives (1) than negatives (0) within the window
        if window.count(1) > window.count(0):
            classes_temporally_voted.append(1) # It is counted as a positive
        else:
            classes_temporally_voted.append(0) # It is counted as a negative
    return classes_temporally_voted

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








def get_table_name(file_path):
    for f in file_path.split(os.sep):
        if f.startswith('table'):
            return f
    return None

# Tentativa de dado um cvs com o alinhamento + folder com features + nome do fold, retorna as features
def get_diff_features(csv_dir, features_dir, name_fold, type_features, layer, types_features, list_consider_tables=None):
    # Create dictionary to gather all features to be returned
    all_diff_features = {'pos': [], 'neg': []}
    X = [] # samples
    Y_hat = [] # labels
    paths_feat = [] # paths of the features
    if type_features == 'object':
        # Pega filtros para todas as features, menos a do objeto alvo
        filters = get_file_filters_train(name_fold, JSON_FILE_OBJECT, layer, include_table_folder=True, types=types_features)
    if type_features == 'research':
        filters = get_file_filters_test(name_fold, JSON_FILE_RESEARCH, layer, include_table_folder=True, types=types_features, list_consider_tables=list_consider_tables)
    # Concat directory with the feature names
    filters = [os.path.join(features_dir,f) for f in filters]
    # Get all pairs ref-tar associated frames listed in all csv files
    dict_features_names = get_all_csv_info(csv_dir)

    # Loop through each filter
    for fil in filters:
        # Get all files based on the filter
        files = glob.glob(fil)
        for file_feature in files:
            # Get info related to the feature based on its path
            info_feature = get_info_dir_path(file_feature)
            table_name = info_feature['table_name']
            object_number = info_feature['object_number']
            feat_type = info_feature['type']
            bn_tar = os.path.basename(file_feature)
            # Find feature in the dictinary that relates all tar-ref pairs
            pos_feats = dict_features_names[table_name]['object_%s'%object_number]['pos']
            neg_feats = dict_features_names[table_name]['object_%s'%object_number]['neg']
            # Get features
            for pos in pos_feats:
                # Check if filename we are looking for is found
                if pos[1] in bn_tar:
                    # Get reference features based on the name of the feature
                    ref_feat_path = glob.glob(features_dir+f'/**/{layer}'+f'/*{pos[0]}', recursive=True)
                    assert len(ref_feat_path) == 1
                    ref_feat_path = ref_feat_path[0]
                    # Get target features based on the name of the feature
                    tar_feat_path = glob.glob(features_dir+f'/**/{layer}'+f'/*{pos[1]}', recursive=True)
                    assert len(tar_feat_path) == 1
                    tar_feat_path = tar_feat_path[0]
                    # Get features
                    feat_map_ref = np.load(ref_feat_path).flatten()
                    feat_map_tar = np.load(tar_feat_path).flatten()
                    diff = feat_map_ref - feat_map_tar
                    all_diff_features['pos'].append(diff)
                    X.append(diff)
                    Y_hat.append(1) # positive class
                    paths_feat.append('[pos] %s - %s'% (os.path.basename(ref_feat_path), os.path.basename(tar_feat_path))) # pair of features
                    break
            for neg in neg_feats:
                # Check if filename we are looking for is found
                    if neg[1] in bn_tar:
                        # Get reference features based on the name of the feature
                        ref_feat_path = glob.glob(features_dir+f'/**/{layer}'+f'/*{neg[0]}', recursive=True)
                        assert len(ref_feat_path) == 1
                        ref_feat_path = ref_feat_path[0]
                        # Get target features based on the name of the feature
                        tar_feat_path = glob.glob(features_dir+f'/**/{layer}'+f'/*{neg[1]}', recursive=True)
                        assert len(tar_feat_path) == 1
                        tar_feat_path = tar_feat_path[0]
                        # Get features
                        feat_map_ref = np.load(ref_feat_path).flatten()
                        feat_map_tar = np.load(tar_feat_path).flatten()
                        diff = feat_map_ref - feat_map_tar
                        all_diff_features['neg'].append(diff)
                        X.append(diff)
                        Y_hat.append(0) # negative class
                        paths_feat.append('[neg] %s - %s'% (os.path.basename(ref_feat_path), os.path.basename(tar_feat_path))) # pair of features
                        break
    return all_diff_features, np.array(X), np.array(Y_hat), paths_feat

def get_info_dir_path(file_path):
    info = {}
    for part in file_path.split(os.sep)[:-1]:
        if 'table_' in part : info['table_name'] = part
        if 'table_' in part : info['table_number'] = part[-2:]
        if '-Object_' in part : info['object_number'] = part[-2:]
        if '-Object_' in part : info['type'] = 'tar'
        if '-Reference_' in part : info['type'] = 'ref'
        for l in layers:
            if part == l : info['layer'] = l
    basename = os.path.basename(file_path)
    if 'path_' in basename : info['path'] = basename[basename.index('path_')+5:basename.index('path_')+6]
    return info

def get_all_csv_info(csv_dir):
    csv_files = glob.glob(csv_dir + '/**/*.csv', recursive=True)
    dict_features = {}
    for csv_path in csv_files:
        # Get information like table nbr and objet nbr of the csv
        info_csv = get_info_dir_path(csv_path)
        table_number = info_csv['table_number']
        object_number = info_csv['object_number']
        path = info_csv['path']
        # Open the csv
        f = open(csv_path)
        csv_file = csv.reader(f, delimiter=',')
        col_ref = 1
        col_tar = 0
        dict_features_table = {'pos': [], 'neg':  []}
        # Loop through the lines
        for row in csv_file:
            if row[0].startswith('ref'):
                col_ref = 0
                col_tar = 1
                continue
            ref_feat = f't{table_number}_ref_vid{object_number}_frame_{row[col_ref]}.npy'
            tar_feat = f't{table_number}_tar_vid{object_number}_frame_{row[col_tar]}.npy'
            # Does not contain annotation -> negative class
            if row[2] == '':
                dict_features_table['neg'].append([ref_feat, tar_feat])
            # Contains annotation -> positive class
            else:
                dict_features_table['pos'].append([ref_feat, tar_feat])
        f.close()
        if 'table_%s'%table_number not in dict_features:
            dict_features['table_%s'%table_number] = {}
        if 'object_%s'%object_number not in dict_features['table_%s'%table_number]:
            dict_features['table_%s'%table_number]['object_%s'%object_number] = {}
        dict_features['table_%s'%table_number]['object_%s'%object_number]['path_%s'%path]= dict_features_table
    return dict_features




n_estimators = 10

layers = ['conv1','residual1','residual2','residual3','residual4','residual5',
          'residual6','residual7','residual8','residual9','residual10','residual11',
          'residual12','residual13','residual14','residual15','residual16']

# type_features = 'object' #ou 'research'
# type_features = 'research' #ou 'object'
# layer = 'conv1'

# types_features = ['tar']
# if type_features == 'object':
#     json_file = JSON_FILE_OBJECT
# elif type_features == 'research':
#     json_file = JSON_FILE_RESEARCH

# Loop through every fold
for fold in folds:
    # Get target object
    tar_obj = folds[fold]
    # Obtain target objects info into the testing structure
    test_object_info = get_objects_info([tar_obj], JSON_FILE_RESEARCH) #AQUI
    # Loop through every layer
    for layer in layers:
        # Calculate TP and FP for the whole layer considering all tables (videos)
        all_TP_no_temporal_voting = 0
        all_FP_no_temporal_voting = 0
        all_DIS_no_temporal_voting = 0
        all_TP_with_temporal_voting = 0
        all_FP_with_temporal_voting = 0
        all_DIS_with_temporal_voting = 0
        # Total number of positive and negative ground truth frames
        total_number_pos_gt_frames = 0
        total_number_neg_gt_frames = 0
        # Print out status for new training
        print('='*40)
        print('\n')
        print('TRAINING')
        print(f'Fold: {fold}')
        print(f'Layer: {layer}')
        # Define path with features for training
        # dir_features_train = os.path.join(features_dir['train'], f'{fold}',f'{layer}')
        # AQUI
        _, X_train, Y_train_hat, paths_feat_training = get_diff_features(csv_dir['train'],features_dir['train'],fold,'object',layer,['tar'])

        # Get features for training
        # X, Y_hat, paths_feat_training = get_features(dir_features_train, max_features_per_class=None, shuffle=True, balance_classes=True)
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
        overall_Y_test_pred_temporal_voting = []
        # Dictionary to save the results for testing
        results['result_testing'] = {}
        # First get all feature samples from each table
        for table in test_object_info:
            print('-'*40)
            print(f'Table: [{table}][{layer}][{tar_obj}]')
            # AQUI -> Para test
            _, X_test, Y_test_hat, paths_feat_testing = get_diff_features(csv_dir['test'],features_dir['test'],fold,'research',layer,['tar'], list_consider_tables=[table])
            # Get features for testing
            # X_test, Y_test_hat, paths_feat_testing = get_features(dir_features_test, max_features_per_class=None, shuffle=False, balance_classes=False)
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
            # Validate the predictions
            final_results, TP, FP = validate_detections(paths_feat_testing, Y_test_pred, Y_test_hat)
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
            print('(DIS): %.2f' % DIS)
            # Metrics with temporal voting for the current layer
            classes_temporally_voted = apply_temporal_voting(Y_test_pred)
            overall_Y_test_pred_temporal_voting += classes_temporally_voted
            final_results_temporal_voting, TP_temporal_voting, FP_temporal_voting = validate_detections(paths_feat_testing, classes_temporally_voted, Y_test_hat)
            print('Temporal voting: (TP, FP): (%d, %d)' % (TP_temporal_voting, FP_temporal_voting))
            # TP_rate: TP / (TP+FN)
            # TP is the number of true positives, FP is the number of false negatives and TP+FN is the total number of positives
            if amount_pos != 0:
                TP_rate_temporal_voting = TP_temporal_voting / amount_pos
            else:
                TP_rate_temporal_voting = 0
            # FP_rate: FP / (FP+TN)
            # FP is the number of false positives, TN is the number of true negatives and FP+TN is the total number of negatives
            if amount_neg != 0:
                FP_rate_temporal_voting = FP_temporal_voting / amount_neg
            else:
                FP_rate_temporal_voting = 0
            print('Temporal voting: (TP rate, FP rate): (%.2f, %.2f)' % (TP_rate_temporal_voting, FP_rate_temporal_voting))
            DIS_temporal_voting = np.sqrt((1-TP_rate_temporal_voting)**2 + FP_rate_temporal_voting**2)
            print('Temporal voting: (DIS): %.2f' % DIS_temporal_voting)

            # Update FP and TP for the currenct layer
            all_TP_no_temporal_voting += TP
            all_FP_no_temporal_voting += FP
            all_TP_with_temporal_voting += TP_temporal_voting
            all_FP_with_temporal_voting += FP_temporal_voting
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
            # Save temporal voting
            results_table[table]['TP_temporal_voting'] = TP_temporal_voting
            results_table[table]['FP_temporal_voting'] = FP_temporal_voting
            results_table[table]['TPR_temporal_voting'] = TP_rate_temporal_voting
            results_table[table]['FPR_temporal_voting'] = FP_rate_temporal_voting
            results_table[table]['DIS_temporal_voting'] = DIS_temporal_voting
            # Add table results to the result_testing dict
            results['result_testing'][table] = results_table[table]
        # Obtain the overall accuracy considering features of all tables
        overall_accuracy = accuracy_score(overall_Y_test_hat, overall_Y_test_pred)
        overall_TP_rate = all_TP_no_temporal_voting/total_number_pos_gt_frames
        overall_FP_rate = all_FP_no_temporal_voting/total_number_neg_gt_frames
        all_DIS_no_temporal_voting = np.sqrt((1-overall_TP_rate)**2 + overall_FP_rate**2)
        results['result_testing']['overall_accuracy'] = overall_accuracy
        results['result_testing']['overall_TP'] = all_TP_no_temporal_voting
        results['result_testing']['overall_FP'] = all_FP_no_temporal_voting
        results['result_testing']['overall_TPR'] = overall_TP_rate
        results['result_testing']['overall_FPR'] = overall_FP_rate
        results['result_testing']['overall_DIS'] = all_DIS_no_temporal_voting
        # overal TP and FP rates for temporal voting
        overall_accuracy_temporal_voting = accuracy_score(overall_Y_test_hat, overall_Y_test_pred_temporal_voting)
        overall_TP_rate_temporal_voting = all_TP_with_temporal_voting/total_number_pos_gt_frames
        overall_FP_rate_temporal_voting = all_FP_with_temporal_voting/total_number_neg_gt_frames
        all_DIS_with_temporal_voting = np.sqrt((1-overall_TP_rate_temporal_voting)**2 + overall_FP_rate_temporal_voting**2)
        results['result_testing']['overall_accuracy_temporal_voting'] = overall_accuracy_temporal_voting
        results['result_testing']['overall_TP_temporal_voting'] = all_TP_with_temporal_voting
        results['result_testing']['overall_FP_temporal_voting'] = all_FP_with_temporal_voting
        results['result_testing']['overall_TPR_temporal_voting'] = overall_TP_rate_temporal_voting
        results['result_testing']['overall_FPR_temporal_voting'] = overall_FP_rate_temporal_voting
        results['result_testing']['overall_DIS_temporal_voting'] = all_DIS_with_temporal_voting
        print('-'*40)
        print('(Overall accuracy): {:.2%}'.format(overall_accuracy))
        print('(Overall TP, FP): (%d, %d)' % (all_TP_no_temporal_voting, all_FP_no_temporal_voting))
        print('(Overall TPR, FPR): (%.2f, %.2f)' % (overall_TP_rate, overall_FP_rate))
        print('(Overall DIS): %.2f' % all_DIS_no_temporal_voting)
        print('Temporal voting: (Overall accuracy): {:.2%}'.format(overall_accuracy_temporal_voting))
        print('Temporal voting: (Overall TP, FP): (%d, %d)' % (all_TP_with_temporal_voting, all_FP_with_temporal_voting))
        print('Temporal voting: (Overall TPR, FPR): (%.2f, %.2f)' % (overall_TP_rate_temporal_voting, overall_FP_rate_temporal_voting))
        print('temporal voting: (Overall DIS): %.2f' % all_DIS_with_temporal_voting)
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
