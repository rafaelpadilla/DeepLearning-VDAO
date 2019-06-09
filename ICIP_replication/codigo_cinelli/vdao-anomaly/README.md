# Moving-camera Video Surveillance in Cluttered Environments using Deep Features
This is code for the paper [Moving-camera Video Surveillance in Cluttered Environments using Deep Features](https://www.researchgate.net/publication/327995320_Moving-Camera_Video_Surveillance_in_Cluttered_Environments_Using_Deep_Features) by Bruno Afonso, Lucas Cinelli, Lucas Thomaz, Allan Silva, Eduardo Barros and Sergio Netto.

## VDAO dataset
To download the ResNet50 features extracted from the [VDAO dataset](http://www02.smt.ufrj.br/%7Etvdigital/database/objects/page_01.html) run:

``` bash
sh datasets/download_vdao.sh /path/to/data/
```

This will download the dataset in hdf5 format into the specified directory.

[This repository](https://github.com/rafaelpadilla/DeepLearning-VDAO/tree/master/VDAO_Access) may be useful for handling VDAO 

## Requirements

 * python 3.6.6
 * hdf5 1.10.2
 * numpy 1.15.0
 * pandas 0.23.4
 * tensorflow 1.3.0
 * tensorflow-gpu 1.3.0
 * keras 2.2.2
 * matplotlib 2.2.3
 * scipy 1.1.0
 * scikit-learn 0.19.1

## Training

To train the model on the VDAO dataset using a fully-connected layer as classifier, run:

``` bash
python main.py --dataset-dir /path/to/data/ --file train_batch_VDAO.h5 -b 32 --save models/mlp --cv-params 'method=leave_one_out' --arch mlp --arch-params 'nb_neurons=[50, 1600]' --optim adamax train --epochs 20 --lr 0.002 --wd 0.005 --val-roc
```

Or to train using a Random Forest as classifier, run:

``` bash
python main.py --dataset-dir /path/to/data/ --file train_batch_VDAO.h5 -b 32 --save models/rf --cv-params 'method=leave_one_out' -arch randomforest --arch-params 'nb_trees=100' --optim adamax train --epochs 20 --lr 0.002 --wd 0.005 --val-roc
```

## Testing

To get the prediction results from the trained model, run:

``` bash
python main.py --dataset-dir /path/to/data/ --file 59_videos_test_batch.h5 --load path/to/your/model  --arch mlp/randomforest --cv-params 'method=leave_one_out' --save test_results predict --optim-thres
```

## Paper results

To obtain the Table 1 of our paper, run:

``` bash
python compute_results.py --file /path/to/test/results --save paper_results --med-filter 5
```

## Citation

If you find this useful for your research, please cite the following paper.

```
@inproceedings{afonso2018anomaly,
   author    = {B. M. Afonso and L. P. Cinelli and L. A. Thomaz and A. F. da Silva and E. A. B. da Silva and S. L. Netto},
   title     = {Moving-camera Video Surveillance in Cluttered Environments using Deep Features},
   booktitle = {2018 25th IEEE International Conference on Image Processing (ICIP)},
   year      = {2018},
   pages     = {2296-2300},
   doi       = {10.1109/ICIP.2018.8451540}
}
```
