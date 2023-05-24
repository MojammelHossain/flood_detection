import yaml
import pathlib
from datetime import datetime




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
