import os,glob
import shutil 
import matplotlib.pyplot as plt
import seaborn as  sns
import pandas as pd
import numpy  as np
from sklearn.metrics import confusion_matrix

from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.imagenet_utils import preprocess_input

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (8, 6)


"""
PlotSamples plots random images from selected category
requires:
selected_category  - selected class label  for plotting int (0-999)
path_images        - path to parent  directory of training images (e.g. ./data/)
cats               - dataframe containing category of images 
saveFig            - if True image will be saved
save_path          - save path of the images
DPI                - dpi quality value for saving images (e.g. 300)
"""

def PlotSamples(selected_category,path_images,cats,save_fig,save_name,save_path,DPI):
    folder_name=cats.loc[selected_category,'folder_name']
    images=glob.glob(os.path.join(path_images,'train',folder_name,'*.jpeg'))
    fig, ax = plt.subplots()
    im = ax.imshow(plt.imread(images[0]));
    ax.axis('off')
    plt.title(cats.loc[selected_category,'Category'])
    if save_fig:
        fig.savefig(os.path.join(save_path,save_name+'.png'),format='png',\
                                   bbox_inches='tight', \
                                   transparent=True,\
                                    dpi=DPI)
        
    plt.show()
    

"""
ArrangeVals moves all validation images into subdirectories\
with the name of the class as directory name
requires:
val_labels           - validation labels dataframe
validation_directory -path to validation images directory
"""
# since all validation images are in one folder we will organize them intp subfolders
def ArrangeVals(val_labels,validation_directory):

    labels=val_labels.labels.unique()
    for im_category in labels:
        try:
            val_cat_folder=os.path.join(validation_directory, im_category)
            if os.path.exists(val_cat_folder)==False:
                os.makedirs(val_cat_folder)
            images=[os.path.join(validation_directory,x) for x in val_labels[val_labels.labels==im_category]['ImageId'].values]

            for im in images:
                shutil.move(im,val_cat_folder)
        except:
            pass
            
 
 
"""
PlotHistory: Utility function to plot learning curves
requires:
history       - Pandas dataframe containing learning loss/accuracy per epochs for both train and validation
modelName     - Model name for plot title
modelSaveName - Model name for saving figure
saveFig       - if True learning curves will be saved
save_path     - save path of the images
showfigs      - if true figures will be shown while training else no figure will be shown
DPI           - dpi quality value for saving images (e.g. 300)
"""
def PlotHistory(history,modelName,modelSaveName,saveFig, save_path, showfigs,DPI):

    hist=pd.DataFrame(history.history)
    hist['epoch']=history.epoch
    fig, (ax1, ax2)=plt.subplots(nrows=1,ncols=2,figsize=(9,6))
    # Loss
    fig.subplots_adjust(hspace=0.5)
    ax1.plot(hist['epoch'],hist['loss'],color='blue',label='Training Loss')
    ax1.plot(hist['epoch'],hist['val_loss'],color='red',label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Model: '+str(modelName),fontsize=15)
    # Accuracy
    ax2.plot(hist['epoch'],hist['accuracy'],color='blue',label='Training Accuracy')
    ax2.plot(hist['epoch'],hist['val_accuracy'],color='red',label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend();
    ax2.set_title('Model: '+str(modelName),fontsize=15)
    if showfigs:
        plt.show();
    if saveFig:
        fig.savefig(save_path+'Learning_Curve_'+str(modelSaveName)+'.png',bbox_inches='tight',format='png',dpi=DPI)
        
"""
PlotConfusionMatrix: Utility function to plot confusion matrix (recommended when selected classes  are less than 5)
requires:
labels        - List of ground truth labels 
predictions   - predicted  labels by  classifier 
saveFig       - if True image will be saved
save_path     - save path of the images
DPI           - dpi quality value for saving images (e.g. 300)
"""       

def PlotConfusionMatrix(labels, predictions,save_fig,save_path,DPI):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    if save_fig:
        plt.savefig(os.path.join(save_path,'Confusion_matrix.png'),bbox_inches='tight',format='png',dpi=DPI)
    plt.show()
    
"""
PlotMisMatch: Utility function to plot images that wrongly classified (only  works if Subset== True)
requires:
datagen                     - data genertator (validation/test)
preds                       - predicted  labels by  classifier 
image_directory             - location of validation/test directory
Subset                      - If  true, only subset  of images  are used to predict
Selected_Classes            - list of selected image categories
cats,save_fig,save_path,DPI - same as other functions 
"""     
def PlotMisMatch(datagen,preds,image_directory,Subset,Selected_Classes,cats,save_fig,save_path,DPI):
    if Subset:
        datagen.shuffle=False
        mismatch=np.nonzero(datagen.classes-preds)[0]
        for im_index in mismatch:
            fig, ax = plt.subplots()
            im = ax.imshow(plt.imread(os.path.join(image_directory,datagen.filenames[im_index])))
            ax.axis('off')
            correct_classId=Selected_Classes[datagen.classes[im_index]]
            correct_class=cats[cats.folder_name==correct_classId]['Category'].values[0]
            pred_classId=Selected_Classes[preds[im_index]]
            pre_class=cats[cats.folder_name==pred_classId]['Category'].values[0]
            plt.title('Correct Label: '+str(correct_class)+'\n'+'Predicted Label: '+str(pre_class))
            if save_fig:
                fig.savefig(os.path.join(save_path,datagen.filenames[im_index].split('\\')[-1].split('.')[0]+'.png'),format='png',\
                                           bbox_inches='tight', \
                                           transparent=True,\
                                            dpi=DPI)
            plt.show()
# following function modify model last activation fucntion
def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    return m

def loss(output):
    return (output[0][0], output[1][1])
