"""
This is a more conventional convolutional eutoencoder where the code
layer is a vector
"""
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D,\
    UpSampling2D, Flatten, Reshape, LeakyReLU, Input, ReLU


MAIN_ACTIVATION_FN = lambda x: x
MAIN_ACTIVATION_FN = LeakyReLU()
SECOND_ACTIVATION_FN = None


class ModelNotFoundError(KeyError):
    """Raise when requesting a model not in model_choices"""
    pass


def model_choices():
    """Alternative model functions to pass to build_model"""
    return {
        'simple': {'encoder': nn_simple_encoder,
                   'autoencoder': nn_simple_autoencoder},
        'plain': {'encoder': nn_v1_encoder,
                  'autoencoder': nn_v1_autoencoder},
        'rgb':{'encoder': rgb_encoder,
               'autoencoder': rgb_autoencoder},
        'rgb_512':{'encoder': rgb_512_encoder,
               'autoencoder': rgb_512_autoencoder},
        'rgb_1024':{'encoder': rgb_1024_encoder,
               'autoencoder': rgb_1024_autoencoder}}

# Current default - flattening autoencoder


def nn_encoder(input_img):

    x = Conv2D(8, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(input_img)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    #x = Conv2D(256, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(512, (1,1), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(512, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(2048, (1,1), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(2048, (2, 2), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2))(x)
    encoded = Flatten()(encoded)
    return encoded


def nn_autoencoder(encoded):
    # the representation is now 1x1x2048
    # fully connected layer
    x = Dense(2048)(encoded)
    #x= MAIN_ACTIVATION_FN(x) #todo - added in on last try
    x = Reshape((1, 1, 2048))(x)

    x = Conv2D(2048, (1,1), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(2048, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(512, (1,1), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(512, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(256, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (5,5), activation='sigmoid', padding='same')(x)
    return decoded






#as above but for rbg images where each layer represents a different filter mask
def rgb_encoder(input_img):
    

    x = Conv2D(8, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(input_img)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    #x = Conv2D(256, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(512, (1,1), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(512, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(2048, (1,1), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(2048, (2, 2), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2))(x)
    encoded = Flatten()(encoded)
    return encoded


def rgb_autoencoder(encoded):
    # the representation is now 1x1x2048
    # fully connected layer
    x = Dense(2048)(encoded)
    #x= MAIN_ACTIVATION_FN(x) #todo - added in on last try
    x = Reshape((1, 1, 2048))(x)

    x = Conv2D(2048, (1,1), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(2048, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(512, (1,1), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(512, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(256, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (5,5), activation='sigmoid', padding='same')(x)
    return decoded




#as above but for rbg images where each layer represents a different filter mask
def rgb_512_encoder(input_img):
    
    x = Conv2D(4, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(input_img)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    

    x = Conv2D(8, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    #x = Conv2D(256, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(512, (1,1), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(512, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(2048, (1,1), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(2048, (2, 2), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2))(x)
    encoded = Flatten()(encoded)
    return encoded


def rgb_512_autoencoder(encoded):
    # the representation is now 1x1x2048
    # fully connected layer
    x = Dense(2048)(encoded)
    #x= MAIN_ACTIVATION_FN(x) #todo - added in on last try
    x = Reshape((1, 1, 2048))(x)

    x = Conv2D(2048, (1,1), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(2048, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(512, (1,1), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(512, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(256, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(4, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    
    
    decoded = Conv2D(3, (5,5), activation='sigmoid', padding='same')(x)
    return decoded






#as above but for rbg images where each layer represents a different filter mask
def rgb_1024_encoder(input_img):
    
    x = Conv2D(2, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(input_img)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(4, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    

    x = Conv2D(8, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    #x = Conv2D(256, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(512, (1,1), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(512, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(2048, (1,1), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(2048, (2, 2), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2))(x)
    encoded = Flatten()(encoded)
    return encoded


def rgb_1024_autoencoder(encoded):
    # the representation is now 1x1x2048
    # fully connected layer
    x = Dense(2048)(encoded)
    #x= MAIN_ACTIVATION_FN(x) #todo - added in on last try
    x = Reshape((1, 1, 2048))(x)

    x = Conv2D(2048, (1,1), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(2048, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(512, (1,1), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(512, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    # x = Conv2D(256, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3,3), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(4, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(2, (5,5), padding='same', activation=SECOND_ACTIVATION_FN)(x)
    x = MAIN_ACTIVATION_FN(x)
    x = UpSampling2D((2, 2))(x)
    
    
    decoded = Conv2D(3, (5,5), activation='sigmoid', padding='same')(x)
    return decoded





# 'plain' choice - original autoencoder prototype
# Doesn't work as-is as input dimensions are different...


def nn_v1_encoder(input_img):
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    return encoded


def nn_v1_autoencoder(encoded):
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    return decoded

# 'simple' choice, minimal model for reference, taken from this article:
# https://towardsdatascience.com/autoencoders-in-keras-c1f57b9a2fd7


def nn_simple_encoder(input_img):
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    return encoded


def nn_simple_autoencoder(encoded):
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    return decoded


def build_model(model_choice=None,
                x_size=None,
                y_size=None,
                z_size=1,
                file_prefix='flattening_',
                loss_func='binary_crossentropy',
                learning_rate=None,
                optimizer=None):
    """Given a keras Input, return encoder and autoencoder models"""
    # TODO pass in optimizer / loss func in a clean way
    opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                                decay=0.0, amsgrad=False)
    # opt=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    # opt=keras.optimizers.SGD(lr=LEARNING_RATE,
    # momentum=0.0, decay=0.05, nesterov=False)
    # Specify a model to use, otherwise the default

    # We define input dimensions here, as re-used for model creatiom
    input_img = Input(shape=(y_size, x_size, z_size))

    if model_choice:
        models = model_choices()
        try:
            encoded = models[model_choice]['encoder'](input_img)
            decoded = models[model_choice]['autoencoder'](encoded)
        except KeyError as err:
            msg = "Could not find a model named {model_choice}"
            raise ModelNotFoundError(msg) from err
    else:
        encoded = nn_encoder(input_img)
        decoded = nn_autoencoder(encoded)

    encoder = Model(input_img, encoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=opt, loss=loss_func)

    return autoencoder, encoder
