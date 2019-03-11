import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_plot_compare_detections(npy_file, table_name=None):
    # Given the npy file with the result of the RF classifier,
    # plot a graph showing the frames with and without objects of a given layer of a particular video

    # Load pickle file
    pkl_file = pickle.load(open(npy_file, "rb"))
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
        title = f'Classification results [table {table_number}] [{fold}] [taget {folds_objects[fold]}] [{layer}]\n[TP:{TP}] [FP:{FP}] [DIS:%.2f] [acc:%s]' % (DIS, acc_perc)
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


def print_accuracy(layer_name, npy_file, table_name=None):
    # Given the npy file with the result of the RF classifier,
    # plot a graph showing the frames with and without objects of a given layer of a particular video

    # Load pickle file
    pkl_file = pickle.load(open(npy_file, "rb"))
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


def print_DIS(layer_name, npy_file, table_name=None):
    # Load pickle file
    pkl_file = pickle.load(open(npy_file, "rb"))
    tables = []
    # If table was not specified, get all tables
    if table_name == None:
        for key in pkl_file['result_testing']:
            if key.startswith('table_'):
                tables.append(key)
    # If table was specified
    else:
        tables.append(table_name)
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

def print_DISs(folds_objects = ['fold_1', 'fold_2','fold_3','fold_4','fold_5','fold_6','fold_7','fold_8','fold_9']):
    for fold in folds_objects:
        for layer in layers_to_generate_plots:
            # Get npy file with results
            npy_file = os.path.join(folder_read_results,fold,f'{layer}.pkl')
            print_DIS(layer, npy_file)

def generate_plots_frames():
    for fold in folds_objects:
        for layer in layers_to_generate_plots:
            # Get npy file with results
            npy_file = os.path.join(folder_read_results,fold,f'{layer}.pkl')
            # Get plot for all tables in the npy file
            plots = get_plot_compare_detections(npy_file)
            # Save
            for tn, plot in plots.items():
                name_file = f'class_results_[{tn}][{fold}][{layer}].png'
                plot.savefig(os.path.join(folder_to_save,name_file))
                print('Plot saved: %s' % os.path.join(folder_to_save,name_file))

def print_accuracies():
    for fold in folds_objects:
        for layer in layers_to_generate_plots:
            # Get npy file with results
            npy_file = os.path.join(folder_read_results,fold,f'{layer}.pkl')
            print_accuracy(layer, npy_file) 

#################################################
#### Generate plot results for each frame  ######
#################################################
# generate_plots_frames()

###########################################################
#### Print accuracies to Ctr+C Ctrl+V in the table   ######
###########################################################
# print_accuracies()

###########################################################
#### Print DIS to Ctr+C Ctrl+V in the table   ######
###########################################################
print_DISs(['fold_1'])
