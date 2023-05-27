import os
import yaml
import torch
import pathlib
import pandas as pd
from metric import MeanIoU
from datetime import datetime
import matplotlib.pyplot as plt


def write_csv(config, metric):
    if os.path.exists((config["csv_log_dir"]+config["csv_log_name"])):
        df = pd.read_csv((config["csv_log_dir"]+config["csv_log_name"]))
        df2 = pd.DataFrame.from_dict([metric])
        df = pd.concat([df, df2])
    else:
        df = pd.DataFrame.from_dict([metric])

    df.to_csv((config["csv_log_dir"]+config["csv_log_name"]), index=False)



def create_mask(pred, mask):
    mask = torch.argmax(mask[0], axis=2).cpu().numpy()
    pred = torch.argmax(torch.permute(pred[0], (1,2,0)), axis=2).detach().cpu().numpy()
    return pred, mask

# Sub-ploting and save
# ----------------------------------------------------------------------------------------------
def display(display_list, idx, directory, score, exp):
    """
    Summary:
        save all images into single figure
    Arguments:
        display_list (dict): a python dictionary key is the title of the figure
        idx (int) : image index in dataset object
        directory (str) : path to save the plot figure
        score (float) : accuracy of the predicted mask
        exp (str): experiment name
    Return:
        save images figure into directory
    """
    plt.figure(figsize=(12, 8))
    title = list(display_list.keys())

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow((display_list[title[i]]), vmin=0, vmax=1)
        plt.axis('off')

    prediction_name = "img_ex_{}_{}_MeanIOU_{:.4f}.png".format(exp, idx, score) # create file name to save
    plt.savefig(os.path.join(directory, prediction_name), bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()



# Save all plot figures
# ----------------------------------------------------------------------------------------------
def show_predictions(dataset, model, config, val=False):
    """
    Summary: 
        save image/images with their mask, pred_mask and accuracy
    Arguments:
        dataset (object): MyDataset class object
        model (object): keras.Model class object
        config (dict): configuration dictionary
        val (bool): for validation plot save
    Output:
        save predicted image/images
    """

    if val:
        directory = config['prediction_val_dir']
    else:
        directory = config['prediction_test_dir']

    # save single image after prediction from dataset
    if config['plot_single']:
        feature, mask, idx = dataset.dataset.get_random_data(config['index'])
        data = [(feature, mask)]
    else:
        data = dataset
        idx = 0

    for feature, mask in data: # save all image prediction in the dataset
        prediction = model(torch.permute(feature, (0,3,1,2)).to("cuda"))

        for i in range(len(feature)): # save single image prediction in the batch
            m = MeanIoU()
            pred = prediction[i].unsqueeze(0)
            msk = mask[i].unsqueeze(0)
            score = m(pred, torch.permute(msk, (0,3,1,2)).to("cuda")).item()
            pred, msk = create_mask(pred, msk)
            display({"Feature": feature[i].numpy(),
                      "Mask": msk,
                      "Prediction (MeanIOU_{:.4f})".format(score): pred
                      }, idx, directory, score, config['experiment'])
            idx += 1


# Model Output Path
# ----------------------------------------------------------------------------------------------
def create_paths(config, test=False):
    """
    Summary:
        creating paths for train and test if not exists
    Arguments:
        config (dict): configuration dictionary
        test (bool): boolean variable for test directory create
    Return:
        create directories
    """
    if test:
        pathlib.Path(config['prediction_test_dir']).mkdir(parents = True, exist_ok = True)
    else:
        pathlib.Path(config['csv_log_dir']).mkdir(parents = True, exist_ok = True)
        pathlib.Path(config['tensorboard_log_dir']).mkdir(parents = True, exist_ok = True)
        pathlib.Path(config['checkpoint_dir']).mkdir(parents = True, exist_ok = True)
        pathlib.Path(config['prediction_val_dir']).mkdir(parents = True, exist_ok = True)

# Create config path
# ----------------------------------------------------------------------------------------------
def get_config_yaml(path, args):
    """
    Summary:
        parsing the config.yaml file and reorganize some variables
    Arguments:
        path (str): config.yaml file directory
        args (dict): dictionary of passing arguments
    Return:
        a dictonary
    """
    with open(path, "r") as f:
      config = yaml.safe_load(f)
    
    # Replace default values with passing values
    for key in args.keys():
        if args[key] != None:
            config[key] = args[key]
            
    config['height'] = config['patch_size']
    config['width'] = config['patch_size']
    
    # Merge paths
    config['train_dir'] = config['dataset_dir']+config['train_dir']
    config['valid_dir'] = config['dataset_dir']+config['valid_dir']
    config['test_dir'] = config['dataset_dir']+config['test_dir']
    
    config['p_train_dir'] = config['dataset_dir']+config['p_train_dir']
    config['p_valid_dir'] = config['dataset_dir']+config['p_valid_dir']
    config['p_test_dir'] = config['dataset_dir']+config['p_test_dir']
    
    # Create Callbacks paths
    config['tensorboard_log_name'] = "{}_ex_{}_epochs_{}_{}".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['tensorboard_log_dir'] = config['root_dir']+'/logs/'+config['model_name']+'/'

    config['csv_log_name'] = "{}_ex_{}_epochs_{}_{}.csv".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['csv_log_dir'] = config['root_dir']+'/csv_logger/'+config['model_name']+'/'

    config['checkpoint_name'] = "{}_ex_{}_epochs_{}_{}.pt".format(config['model_name'],config['experiment'],config['epochs'],datetime.now().strftime("%d-%b-%y"))
    config['checkpoint_dir'] = config['root_dir']+'/model/'+config['model_name']+'/'

    # Create save model directory
    if config['load_model_dir']=='None':
        config['load_model_dir'] = config['root_dir']+'/model/'+config['model_name']+'/'
    
    # Create Evaluation directory
    config['prediction_test_dir'] = config['root_dir']+'/prediction/'+config['model_name']+'/test/'
    config['prediction_val_dir'] = config['root_dir']+'/prediction/'+config['model_name']+'/validation/'

    return config
