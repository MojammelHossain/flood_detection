import segmentation_models as sm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Conv2D, Input, MaxPooling2D, concatenate, Conv2DTranspose, Dropout


# UNET Model
# ----------------------------------------------------------------------------------------------

def unet(config):
    
    """
        Summary:
            Create UNET model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    
    inputs = Input((config['height'], config['width'], config['in_channels']))
 
    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.2)(c1)  # Original 0.1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)  # Original 0.1
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)  # Original 0.1
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(config['num_classes'], (1, 1), activation='softmax', dtype='float32')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


def linknet(config):
    """
        Summary:
            Create LINKNET from segmentation models library model object
        Arguments: 
            Model configuration from config.yaml
        Return:
            Keras.model object
    """

    model = sm.Linknet(backbone_name='efficientnetb0', input_shape=(config['height'], config['width'], config['in_channels']),
                    classes = config['num_classes'], activation='softmax',
                    encoder_weights=None, weights=None)
    x = model.layers[-2].output # fetch the last layer previous layer output
    
    output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
    model = Model(inputs = model.input, outputs=output)
    return model

# Transfer Learning
# ----------------------------------------------------------------------------------------------

def get_model_transfer_lr(model, num_classes):
    """
    Summary:
        create new model object for transfer learning
    Arguments:
        model (object): keras.Model class object
        num_classes (int): number of class
    Return:
        model (object): keras.Model class object
    """


    x = model.layers[-2].output # fetch the last layer previous layer output
    
    output = Conv2D(num_classes, kernel_size = (1,1), name="out", activation = 'softmax')(x) # create new last layer
    model = Model(inputs = model.input, outputs=output) 
    
    # freeze all model layer except last layer
    for layer in model.layers[:-1]:
        layer.trainable = False
    
    return model

# Get model
# ----------------------------------------------------------------------------------------------

def get_model(config):
    """
    Summary:
        create new model object for training
    Arguments:
        config (dict): Configuration directory
    Return:
        model (object): keras.Model class object
    """


    models = {'unet': unet,
              'linknet': linknet
              }
    return models[config['model_name']](config)