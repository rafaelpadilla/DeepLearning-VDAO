import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_plot_compare_detections(pkl_file, fold_name, layer_name, table_name=None):
    # Given the file with the result of the RF classifier,
    # plot a graph showing the frames with and without objects of a given layer of a particular video
    fold_number = l_double_digit(fold_name.replace('fold_',''))
    # Load pickle file
    pkl_file = pickle.load(open(pkl_file, "rb"))
    tables = []
    # If table was not specified, get all tables
    if table_name == None:
        for key in pkl_file['result_testing']:
            if key.startswith('table_'):
                tables.append(key)
    # If table was specified
    else:
        tables.append(table_name)
    ret_plots = { }
    # Loop through tables to generate plot
    for tn in tables:
        table_number = l_double_digit(tn.replace('table_',''))
        table_info = pkl_file['result_testing'][tn]
        # Get values to be added in the title
        accuracy = table_info['accuracy']
        acc_perc = '{:.2f}%'.format(accuracy*100)
        TP = table_info['TP']
        FP = table_info['FP']
        DIS = table_info['DIS']
        title = f'Classification results [table {table_number}] [fold {fold_number}] [target {folds_objects[fold_name]}] [{layer_name}]\n[TP:{TP}] [FP:{FP}] [DIS:%.2f] [acc:%s]' % (DIS, acc_perc)
        # Get results with information of the gt and detections 
        summary_results = table_info['summary_results']
        gt_and_detections = separate_gts_and_detections(summary_results)
        # Create plot
        fig, ax = plt.subplots()
        ax.plot(gt_and_detections['frames_order'], gt_and_detections['pred_classes'], '.', color='blue', label='detected classes')
        ax.plot(gt_and_detections['frames_order'], gt_and_detections['gt_classes'], color='green', label='ground truth classes')
        ax.set(xlabel='frame number', ylabel='class',title=title)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
        ax.grid()
        # plt.show()
        ret_plots[tn] = fig
    return ret_plots


def print_accuracy(layer_name, pkl_file, table_name=None):
    # Given the file with the result of the RF classifier,
    # plot a graph showing the frames with and without objects of a given layer of a particular video

    # Load pickle file
    pkl_file = pickle.load(open(pkl_file, "rb"))
    tables = []
    # If table was not specified, get all tables
    if table_name == None:
        for key in pkl_file['result_testing']:
            if key.startswith('table_'):
                tables.append(key)
    # If table was specified
    else:
        tables.append(table_name)
    accuracies = ''
    # Loop through tables to generate plot
    for tn in tables:
        table_info = pkl_file['result_testing'][tn]
        # Get values to be added in the title
        accuracy = table_info['accuracy']
        if accuracies == '':
            accuracies = '%s\t%f'%(layer_name,accuracy)
        else:
            accuracies = '%s\t\t%f' % (accuracies, accuracy)
    accuracies = '%s\t\t%f' % (accuracies, pkl_file['result_testing']['overall_accuracy'])
    print(accuracies.replace('.',','))


def print_DIS(layer_name, pkl_file, table_name=None, temporal_voting=False):
    # Load pickle file
    pkl_file = pickle.load(open(pkl_file, "rb"))
    tables = []
    # If table was not specified, get all tables
    if table_name == None:
        for key in pkl_file['result_testing']:
            if key.startswith('table_'):
                tables.append(key)
    # If table was specified
    else:
        tables.append(table_name)
    # Overall metrics
    if temporal_voting:
        overall_TP = pkl_file['result_testing']['overall_TP_temporal_voting']
        overall_TPR = pkl_file['result_testing']['overall_TPR_temporal_voting']
        overall_FP = pkl_file['result_testing']['overall_FP_temporal_voting']
        overall_FPR = pkl_file['result_testing']['overall_FPR_temporal_voting']
        overall_DIS = pkl_file['result_testing']['overall_DIS_temporal_voting']
    else:
        overall_TP = pkl_file['result_testing']['overall_TP']
        overall_TPR = pkl_file['result_testing']['overall_TPR']
        overall_FP = pkl_file['result_testing']['overall_FP']
        overall_FPR = pkl_file['result_testing']['overall_FPR']
        overall_DIS = pkl_file['result_testing']['overall_DIS']
    linha = ''
    # Loop through tables to generate plot
    for tn in tables:
        table_info = pkl_file['result_testing'][tn]
        # Get values to be added in the title
        if temporal_voting:
            TP = table_info['TP_temporal_voting']
            TPR = table_info['TPR_temporal_voting']
            FP = table_info['FP_temporal_voting']
            FPR = table_info['FPR_temporal_voting']
            DIS = table_info['DIS_temporal_voting']
        else:
            TP = table_info['TP']
            TPR = table_info['TPR']
            FP = table_info['FP']
            FPR = table_info['FPR']
            DIS = table_info['DIS']
        if linha == '':
            linha = '%s\t%s\t%s\t%s\t%s\t%s' % (layer_name,TP,TPR,FP,FPR,DIS)
        else:
            linha = '%s\t%s\t%s\t%s\t%s\t%s' % (linha,TP,TPR,FP,FPR,DIS)
    linha = '%s\t%s\t%s\t%s\t%s\t%s' % (linha,overall_TP,overall_TPR,overall_FP,overall_FPR,overall_DIS)
    print(linha.replace('.',','))

# Lambda to insert "0" in 1-digit numbers (eg: 4->"04")
l_double_digit = lambda x : '0'+str(x) if len(str(x)) == 1 else str(x)

def get_frame_number(frame_path):
    frame_path = frame_path.replace('_ann','')
    return int(frame_path[frame_path.rfind('_')+1:].replace('frame','').replace('.npy',''))

def separate_gts_and_detections(summary_results):
    frame_orders = []
    gt_classes = []
    pred_classes = []
    for key, value in summary_results.items():
        frame_orders.append(get_frame_number(key))
        gt_classes.append(value['groundtruth_class'])
        pred_classes.append(value['predicted_class'])
    # Get indices of the sorted frame order
    idx  = np.argsort(frame_orders)
    # Reorganize the lists based on the indexes
    frame_orders = list(np.array(frame_orders)[idx])
    gt_classes = list(np.array(gt_classes)[idx])
    pred_classes = list(np.array(pred_classes)[idx])
    return {'frames_order': frame_orders, 'gt_classes': gt_classes, 'pred_classes': pred_classes}


####################################################################################################
# Parameters to change (folder names)
####################################################################################################
folder_read_results = './RF_results/'
folder_to_save = './RF_results/figures/'
if not os.path.isdir(folder_to_save):
    os.makedirs(folder_to_save)



target_objects = ['shoe', 'towel', 'brown box', 'black coat', 'black backpack', 'dark blue box', 'camera box', 'white jar', 'pink bottle']
folds_objects = {'fold_1': target_objects[0],
         'fold_2': target_objects[1],
         'fold_3': target_objects[2],
         'fold_4': target_objects[3],
         'fold_5': target_objects[4],
         'fold_6': target_objects[5],
         'fold_7': target_objects[6],
         'fold_8': target_objects[7],
         'fold_9': target_objects[8]
}
layers_to_generate_plots = ['conv1','residual1','residual2','residual3','residual4','residual5',
          'residual6','residual7','residual8','residual9','residual10','residual11',
          'residual12','residual13','residual14','residual15','residual16']

def print_DISs(folds_names = ['fold_1', 'fold_2','fold_3','fold_4','fold_5','fold_6','fold_7','fold_8','fold_9'], temporal_voting=False):
    # Print both with temporal and without temporal
    if temporal_voting == None:
        par = [False,True]
    # Print either with temporal voting or without it
    else: 
        par = [temporal_voting]
    for p in par:
        if p == True:
            print('\nWith temporal voting:')
        else:
            print('\nWithout temporal voting:')
        for fold in folds_names:
            print(f'Fold: {fold}')
            for layer in layers_to_generate_plots:
                # Get npy file with results
                pkl_file = os.path.join(folder_read_results,fold,f'{layer}.pkl')
                print_DIS(layer, pkl_file, temporal_voting=p)


def generate_plots_frames(folds_names = ['fold_1', 'fold_2','fold_3','fold_4','fold_5','fold_6','fold_7','fold_8','fold_9']):
    for fold in folds_names:
        for layer in layers_to_generate_plots:
            # Get npy file with results
            pkl_file = os.path.join(folder_read_results,fold,f'{layer}.pkl')
            # Get plot for all tables in the npy file
            plots = get_plot_compare_detections(pkl_file, fold, layer)
            # Save
            for tn, plot in plots.items():
                name_file = f'class_results_[{tn}][{fold}][{layer}].png'
                plot.savefig(os.path.join(folder_to_save,name_file))
                print('Plot saved: %s' % os.path.join(folder_to_save,name_file))

def print_accuracies(folds_names = ['fold_1', 'fold_2','fold_3','fold_4','fold_5','fold_6','fold_7','fold_8','fold_9']):
    for fold in folds_names:
        for layer in layers_to_generate_plots:
            # Get npy file with results
            npy_file = os.path.join(folder_read_results,fold,f'{layer}.pkl')
            print_accuracy(layer, npy_file) 


def generate_plots_results(folds_names=['fold_1', 'fold_2','fold_3','fold_4','fold_5','fold_6','fold_7','fold_8','fold_9']):
    dic = {'DIS': 'overall_DIS',
        'accuracy': 'overall_accuracy'}
    # Create plots for each folder
    for d in dic:
        plots = get_plots_by_folds(folds_names,metric_tag=dic[d])
        for func_name, plot in plots.items():
            name_file = f'{func_name}_{d}_results.png'
            plot.savefig(os.path.join(folder_to_save,name_file))
            print('Plot saved: %s' % os.path.join(folder_to_save,name_file))
    # Create plots for functions max, min and average among all folds
    for d in dic:
        plots = get_plots_among_all_folds(folds_names,metric_tag=dic[d])
        for func_name, plot in plots.items():
            name_file = f'{func_name}_{d}_results.png'
            plot.savefig(os.path.join(folder_to_save,name_file))
            print('Plot saved: %s' % os.path.join(folder_to_save,name_file))

            
def get_plots_by_folds(folds_names, metric_tag='overall_accuracy'):
# metric_tag='overall_accuracy' or metric_tag='overall_DIS'
    if metric_tag == 'overall_accuracy':
        title_metric = 'Accuracy'
        label_metric = 'accuracies'
    elif metric_tag == 'overall_DIS':
        title_metric = 'DIS'
        label_metric = 'DIS'
    else:
        raise Exception('metric_tag must be either \'overall_accuracy\' or \'overall_DIS\'')
    ret_plots = {}
    for fold in folds_names:
        fold_number = l_double_digit(fold.replace('fold_',''))
        x = []
        y = []
        for layer in layers_to_generate_plots:
            # Get file with results
            pkl_file = os.path.join(folder_read_results,fold,f'{layer}.pkl')
            # Get overall accuracies in the npy file
            pkl_file = pickle.load(open(pkl_file, "rb"))
            best_accuracy_layer = pkl_file['result_testing'][metric_tag]
            x.append(layer.replace('residual','res'))
            y.append(best_accuracy_layer)
        # Create plot
        fig, ax = plt.subplots()
        title = f'{title_metric} [fold {fold_number}] [target {folds_objects[fold]}]'
        ax.plot(x, y, '-ob', label=f'{label_metric}')
        ax.plot(x, len(y)*[np.average(y)], color='r', label='average', linestyle='--')
        plt.xticks(rotation=90)
        ax.set(xlabel='layer', ylabel=f'{title_metric}',title=title)
        ax.legend(loc='best', fancybox=True, shadow=True, ncol=2)
        ax.grid()
        # plt.show()
        ret_plots[fold] = fig
    return ret_plots


def get_plots_among_all_folds(folds_names,metric_tag='overall_accuracy'):
# metric_tag='overall_accuracy' or metric_tag='overall_DIS'
    if metric_tag == 'overall_accuracy':
        title_metric = 'accuracy'
        label_metric = 'accuracies'
    elif metric_tag == 'overall_DIS':
        title_metric = 'DIS'
        label_metric = 'DIS'
    else:
        raise Exception('metric_tag must be either \'overall_accuracy\' or \'overall_DIS\'')
    functions = {'max':np.max,'min':np.min,'avg':np.average}
    # Dictionary with all folds
    dict_all_folds = {}
    for fold in folds_names:
        for layer in layers_to_generate_plots:
            # Get file with results
            pkl_file = os.path.join(folder_read_results,fold,f'{layer}.pkl')
            # Get overall accuracies in the npy file
            pkl_file = pickle.load(open(pkl_file, "rb"))
            best_accuracy_layer = pkl_file['result_testing'][metric_tag]
            name_layer = layer.replace('residual','res')
            if name_layer not in dict_all_folds:
                dict_all_folds[name_layer] = [best_accuracy_layer]
            else:
                dict_all_folds[name_layer].append(best_accuracy_layer)
    # Loop trough all results of all folds and get all functions results per layer
    results_functions = {}
    for func_name, func in functions.items():
        # Results
        results_functions[func_name] = {}
        for f, l in dict_all_folds.items():
            results_functions[func_name][f] = func(l)
    ret_plots = {}
    # Go through all functions and create plot
    for func_name in results_functions:
        x = list(results_functions[func_name].keys())
        y = []
        for l in x:
            y.append(results_functions[func_name][l])
        # Create plot
        fig, ax = plt.subplots()
        func_name = func_name[0].upper()+func_name[1:]
        title = f'{func_name} {title_metric} among all folds per layer'
        ax.plot(x, y, '-ob', label=f'{label_metric}')
        # ax.plot(x, len(y)*[np.average(y)], color='r', label='average', linestyle='--')
        plt.xticks(rotation=90)
        ax.set(xlabel='layer', ylabel=f'{title_metric}',title=title)
        ax.legend(loc='best', fancybox=True, shadow=True, ncol=2)
        ax.grid()
        # plt.show()
        ret_plots[func_name] = fig
    return ret_plots

#################################################
#### Generate plot results for each frame  ######
#################################################
# generate_plots_frames()


#################################################
#### Generate plot results for the folds   ######
#################################################
# generate_plots_results()


###########################################################
#### Print accuracies to Ctr+C Ctrl+V in the table   ######
###########################################################
# print_accuracies(['fold_1'])

###########################################################
#### Print DIS to Ctr+C Ctrl+V in the table   ######
###########################################################
print_DISs(['fold_3'], temporal_voting=False)

