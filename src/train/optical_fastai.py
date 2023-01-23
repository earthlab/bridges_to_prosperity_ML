import geopandas as gpd
import numpy as np
import os
from fastai.vision.all import *
from glob import glob
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from IPython.display import display
from matplotlib import pyplot as plt
from shapely import Polygon
import shutil
from datetime import date

def _fastai_format_inputs(
    csv_files: dict, 
    tiff_dirs: dict
    ):
    # obtain the region names from the dictionary of csv files
    regions = csv_files.keys()

    # initialize empty lists that will store the values needed to create a dataframe
    region_name = []
    file_names = []
    labels = []

    # iterate over all the regions for which we have training data
    for region in regions:
        # read the csv file of bridge locations for the current region into a dataframe
        train_csv = pd.read_csv(csv_files[region])

        # create a geopandas POINT geometry for each of the bridge locations in the current csv file
        # the x-coordinate is the longitude value in the dataframe and the y-coordinate is the latitude value in the dataframe 
        train_geometry = gpd.points_from_xy(train_csv['Longitude'], train_csv['Latitude'])

        # obtain the path to the directory containing tiff tiles over the current region
        train_dir = y[region]

        # iterate over all the tiff tiles in the directory containing tiff tiles for the current region
        for tiff in os.listdir(train_dir):
            # save the name of the region associated with the current tiff tile
            region_name.append(region)
            # save the name of the current tiff tile
            file_names.append(tiff)

            tiff_checks = []

            # iterate over all of the verified bridge locations for the current region and check whether any of the bridge locations are located inside the bounding box of the current tiff tile
            for geom in train_geometry:
                # find the bounding box of the current tiff tile
                tiff_bound = tiff.unary_union.bounds
                tiff_bound = Polygon(
                    (
                        (tiff_bound[0], tiff_bound[1]), 
                        (tiff_bound[0], tiff_bound[3]), 
                        (tiff_bound[2], tiff_bound[3]), 
                        (tiff_bound[2], tiff_bound[1])
                    )
                )
                
                # check if the current tiff tile contains the current verified bridge location
                if tiff_bound.contains(geom):
                    tiff_checks.append('yes')
                else:
                    tiff_checks.append('no')

            # label the current tiff tile as Yes if it contains any of the verified bridge locations in this region and label the tiff tile No otherwise
            if np.any(tiff_checks == 'yes'):
                labels.append('yes')
            else:
                labels.append('no')

    # create a dataframe for all the tiff tiles that provides the region for the tiff tile, the name of the tiff tile, and the label for the tiff tile
    df = pd.DataFrame(
        {
            'Region': region_name, 
            'Image Path': file_names, 
            'Label': labels
        }
    )

    # find the indices in the dataframe where the label is Yes
    yes_labels = np.where(df['Label'] == 'yes')[0]
    # find the indices in the dataframe where the label is No
    no_labels = np.where(df['Label'] == 'no')[0]

    # randomly select 70% of the Yes indices to be in the training set
    yes_train_indx = np.random.choice(yes_labels, size = 0.7*len(yes_labels), replace = False)
    # randomly select 70% of the No indices to be in the training set
    no_train_indx = np.random.choice(no_labels, size = 0.7*len(no_labels), replace = False)
    # note that we select 70% of both the Yes and No tiles in order to stratify the training data by label
    # combine the indices of the Yes and No training data to obtain a numpy array containing all the indices that will be in the training set
    train_indx = np.concatenate([yes_train_indx, no_train_indx])

    # select the training set from the dataframe, then shuffle the values and drop their index value
    df_train = df.iloc[train_indx,:].copy()
    df_train = df_train.sample(frac = 1).reset_index(drop = True)

    # obtain the indices of the remaining indices from both the Yes and No labels, which will comprise the validation set
    yes_val_indx = [i for i in yes_labels if i not in yes_train_indx]
    no_val_indx = [i for i in no_labels if i not in no_train_indx]
    # combine the Yes and No indices that will be in the validation set to obtain a list of all the indices in the validation set
    val_indx = yes_val_indx + no_val_indx

    # select the validation set from the dataframe, then shuffle the values and drop their index value
    df_val = df.iloc[val_indx,:].copy()
    df_val = df_val.sample(frac = 1).reset_index(drop = True) 

    # we now define paths to folders that will store the training and validation tiff tiles, respectively
    training_dir = os.get_cwd() + '/training_data'
    val_dir = os.get_cwd() + '/validation_data'

    # we will now make the folders that will store the training and validation tiff tiles if they do not already exist
    os.mkdir(training_dir)
    os.mkdir(val_dir)

    # create two lists that will store the paths to each of the training and validation tiffs, respectively
    training_paths = []
    validation_paths = []

    # iterate over all the indices in the original dataframe
    for i in range(len(df['Region'])):
        # obtain the region and filename
        region = df.iloc[i, 0]
        file = df.iloc[i, 1]

        # if the current tiff belongs to the training set, copy it to the training folder and store its new path; if it belongs to the validation set, copy it to the validation folder and save its new path
        if i in train_indx:
            training_path = training_dir + '/' + file
            training_paths.append(training_path)
            shutil.copy(tiff_dirs[region] + '/' + file, training_path)
        else:
            validation_path = val_dir + '/' + file
            validation_paths.append(validation_path)
            shutil.copy(tiff_dirs[region] + '/' + file, validation_path)
    
    # create a new dataframe containing the paths to the training tiff tiles and their labels
    train_df = pd.DataFrame({'image_file': training_paths, 'image_label': df_train['Label']})

    # create a new dataframe containing the paths to the validation tiff tiles and their labels
    val_df = pd.DataFrame({'image_file': validation_paths, 'image_label': df_val['Label']})

    return train_df, val_df

def _fastai_train_optical(
    train_df,
    batch_sz,
    resnt,
    model_dir
):

    # define several image transformations to augment the training tiff tiles
    tfms = get_transforms(
        flip_vert = True, 
        max_lighting = 0.2, 
        max_zoom = 1.05, 
        max_warp = 0.
    )

    # create a labeled image list for the training tiff tiles
    src = (ImageList.from_df(train_df, path='/')).split_by_rand_pct(0.1).label_from_df('image_label')

    # augment the training tiff tiles using the previously defined transformations
    data = (src.transform(tfms, size=224).databunch().normalize(imagenet_stats))
    # define the batch size
    data.batch_size = batch_sz


    # we will now train all the possible RESNET models using the augmented training tiff tiles
    interps=[]

    # iterate over all the possible RESNET models, select each RESNET model, and perform training
    for lyr in resnt:
        if lyr==50:
            arch = models.resnet50
        elif lyr==152:
            arch = models.resnet152
        elif lyr==34:
            arch = models.resnet34
        elif lyr==18:
            arch = models.resnet18
        elif lyr==101:
            arch = models.resnet101
        else:
            raise Exception(f'{lyr} not implemented')

        # define a learner that will train the current RESNET model on the augmented training data
        acc_02 = partial(
            accuracy_thresh, 
            thresh=0.2
        )
        f_score = partial(
            fbeta, 
            thresh=0.2, 
            beta=1, 
            sigmoid=False
        )
        learn = vision_learner(
            data, 
            arch, 
            metrics=error_rate, 
            ps = 0.75, 
            bn_final=True
        )

        # use the LR Finder to pick a good learning rate.
        learn.lr_find()
        learn.recorder.plot()
  
        # fit the head of our network.
        lr = 0.015
        data.one_batch()[1].shape, data.one_batch()[0].shape
  
        learn.fit(5, slice(lr))

        # we now fine-tune the whole model
        learn.unfreeze()
        learn.lr_find()
        learn.recorder.plot()
        learn.fit_one_cycle(10, slice(1e-3, lr/5))
        learn.recorder.plot_losses()

        # evaluate the performance of the trained RESNET model
        interp = ClassificationInterpretation.from_learner(learn)
        
        # save the performance evaluation for the current RESNET model
        interps.append(interp)

        # save results to file 
        model_name = f'resnet{lyr[i]}_300px_r0-2500_g0-2500_b0-2500_v{date.today()}.pkl'
        learn.export(
            os.path.join(model_dir, model_name)
        )

    return interps
def _fastai_metrics(val_df, interps, resnt):

    ## Evalutat over training data 
    # We will now examine more metrics to evaluate the performance of the models
    precisions, recalls, f1s = [], [], []
    # iterate over all the trained RESNET models from before
    for ex in interps:

        ex.plot_confusion_matrix(normalize=True, cmap='jet_r')
        ex.plot_confusion_matrix(normalize=False, cmap='jet_r')
        ex.confusion_matrix()
        ex.plot_top_losses(9)
        # obtain the confusion matrix for the current RESNET model
        cm = ex.confusion_matrix()
        num_classes=2

        # find the true positive, false positive, false negative, and true negative rates for the current RESNET model
        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP
        TN = []
        for i in range(num_classes):
            temp = np.delete(cm, i, 0)    # delete ith row
            temp = np.delete(temp, i, 1)  # delete ith column
            TN.append(sum(sum(temp)))

        # find the precision and recall of the current RESNET model and save these values
        precision = TP/(TP+FP)
        precisions.append(precision)
        recall = TP/(TP+FN)
        recalls.append(recall)

        # find the F1 score for the current RESNET model and save its value
        f1 = 2*(precision*recall)/(precision+recall)
        f1s.append(f1)

    # create a dataframe of the results for each RESNET model that was trained previously
    results_df = pd.DataFrame({'num_layers': (18,34,50,101,152), 'f1_class_no': np.array(f1s)[:,0], 'f1_class_yes': np.array(f1s)[:,1], 'precision': precisions, 'recall': recalls}) 

    # create a list that specifies the architecture of the RESNET models that we trained previously

    resnet_results = []

    # iterate over all the RESNET models that we trained previously
    for z in resnt:
        # load the saved state of the current RESNET model from its pkl file
        learn_infer = load_learner(path = model_dir, file = f'resnet{z}_300px_r0-2500_g0-2500_b0-2500_v{date.today()}.pkl')

        # define a set of image transforms that will be used to augment the validation data
        tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
        # set the seed to make the results reproducible across multiple runs
        np.random.seed(42)

        # create an image list from the validation tiff tiles and use the transformations defined previously to augment this data
        test_data = ImageList.from_df(val_df, path = '/').split_none().label_empty().transform(tfms, size=224, tfm_y=False).databunch().normalize(imagenet_stats)
        test_data.train_dl.new(shuffle=False)

        # we will use the val_dict to convert from a 0 or 1 value that is output by the model to a string label
        val_dict = {1:'yes', 0:'no'}
        # create empty lists to store the validation results from the current RESNET model
        labels, values, fl_vals = [], [], []

        # iterate over all the validation tiffs
        for i in range(len(val_df['image_label'])):
            # select the current validation tiff
            im = test_data.train_ds[i][0] 
            # use the current RESNET model to predict the current validation tiff
            prediction = learn_infer.predict(im)

            # select the predicted probability for the yes class from the model prediction
            pred_val = prediction[1].data.item()
            # convert the predicted probability to a predicted value using the dictionary from above
            pred_label = val_dict[pred_val] 
            # select the fl value from the model prediction
            fl_val = prediction[2].data[pred_val].item()

            # save the predicted label, predicted probability, and predicted fl value from the current RESNET model for the current validation tiff
            labels.append(pred_label)
            values.append(pred_val) 
            fl_vals.append(fl_val)
  
        # create a dataframe containing the predicted labels, predicted probabilities, and predicted fl values for all the validation tiffs
        results_df = pd.DataFrame({'pred_label': labels, 'pred_value': values, 'fl_val': fl_vals})

        # save the dataframe of validation results for the current RESNET model
        resnet_results.append(results_df)

    clf_reports = []
    cf_mats = []
    tpr_ls = []
    fpr_ls = []
    auc_ls = []

    # iterate over all the trained RESNET models
    for i in resnt:
        # obtain the dataframe containing the validation results for the current RESNET model
        results = resnet_results[i]

        # select the true image labels for the validation tiffs
        true = val_df['image_label']

        # convert the true image labels for the validation tiffs to binary labels, where 1 encodes a yes and 0 encodes a no and add them to the validation dataframe
        val_df['binary_label'] = (val_df['image_label'] == 'yes').astype('int64')
        # select the binary image labels for the validation tiffs
        true_binary = val_df['binary_label']

        # select the predicted labels for the validation tiffs from the current RESNET model
        pred = results['pred_label']
        # select the predicted probabilities for the validation tiffs from the current RESNET model
        pred_prob = results['pred_value']

        # calculate classification report and confusion matrix, as well as TN/FP/FN/TP for the validation performance of the current RESNET model
        clf_report = classification_report(true, pred, output_dict=True)
        cf_mat = confusion_matrix(true, pred)
        tn, fp, fn, tp = cf_mat.ravel()

        # save the classification report and confusion matrix for the validation performance of the current RESNET model
        clf_reports.append(clf_report)
        cf_mats.append(cf_mat)

        # ROC Curves
        # obtain the false positive rate, true positive rate, and ROC AUC values for the validation performance of the current RESNET model
        fpr, tpr, _ = roc_curve(true_binary, pred_prob)
        roc_auc = auc(fpr, tpr)

        # save the true positive rate, false positive rate, and ROC AUC for the validation performance of the current RESNET model
        tpr_ls.append(tpr)
        fpr_ls.append(fpr)
        auc_ls.append(roc_auc)

    # iterate over all the trained RESNET models and their corresponding classification reports and print the results for each RESNET model
    for lyrs, report in zip(resnt, clf_reports):
        print(f'RESNET-{lyrs} Validation Results')

        display(pd.DataFrame(report).T.style.highlight_max(color = 'lightgreen', axis = 0))

        print()
        print()

    # iterate over all the trained RESNET models, the true positive rates, the false positive rates, and the ROC AUC values and create and show a plot of the results for each RESNET model
    for lyrs, _t, _f, _auc in zip(resnt, tpr_ls, fpr_ls, auc_ls):
        print(f'RESNET-{lyrs} Validation Results')
  
        plt.figure()
        lw = 2
        plt.plot(_f, _t, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % _auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic - Uganda')
        plt.legend(loc="lower right")
        plt.show()

        print()
        print()
    return None