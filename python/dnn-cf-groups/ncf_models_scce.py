from data_utils import init_random

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Embedding, Input, Dense, Reshape, Flatten, Concatenate, Multiply
import numpy as np
import matplotlib.pyplot as plt


def mlp_2(model_name, k, dataset, seed, activation_func):
    ds_shape, nuser, nitems, ds_code = dataset.get_shape(), dataset.get_num_users(), dataset.get_num_users(), dataset.get_data_code()
    # Do not put a dataset function call in lambda inside a funciton(mlp_2), keras get a reference and try to serialize it when the model is saved.
    # Very weird situation, dificult to debug. It works in the main file but not when put inside a function
    
    input_layer = layers.Input(shape=ds_shape, name="entrada")

    e_user = layers.Dense(
        k,
        name="emb_u",
        use_bias=False,
    )(layers.Lambda(lambda x: x[:, 0:nuser])(input_layer))
    
    e_item = layers.Dense(
        k,
        name="emb_i",
        use_bias=False,
    )(layers.Lambda(lambda x: x[:, nitems:])(input_layer))
    
    x_u = layers.Dense(128, activation=activation_func)(e_user)
    x_u = layers.Dropout(0.1)(x_u)
    x_u = layers.Dense(64, activation=activation_func)(x_u)
    x_u = layers.Dropout(0.1)(x_u)
    x_u = layers.Dense(32, activation=activation_func)(x_u)
    
    x_i = layers.Dense(128, activation=activation_func)(e_item)
    x_i = layers.Dropout(0.1)(x_i)
    x_i = layers.Dense(64, activation=activation_func)(x_i)
    x_i = layers.Dropout(0.1)(x_i)
    x_i = layers.Dense(32, activation=activation_func)(x_i)
    
    x_embedding = layers.Concatenate(axis=1)([x_u, x_i])
    x_embedding = layers.Dense(16, activation=activation_func)(x_embedding)
    #x_embedding = layers.Dropout(0.1)(x_embedding)
    #x_embedding = layers.Dense(4, activation=activation_func)(x_embedding)
    outputs = layers.Dense(5,activation="softmax")(x_embedding)
    model = keras.Model(inputs=input_layer, outputs=outputs, name=f"{model_name}_{k}_{ds_code}_{seed}")
    model.summary()
    
    return model


def mlp_2_relu(model, k, dataset, seed):
    return mlp_2(model, k, dataset, seed, 'relu')

def mlp_2_tanh(model, k, dataset, seed):
    return mlp_2(model, k, dataset, seed, 'tanh')

def mlp_2_linear(model, k, dataset, seed):
    return mlp_2(model, k, dataset, seed, 'linear')

def mlp_2_lrelu(model, k, dataset, seed):
    return mlp_2(model, k, dataset, seed, tf.keras.layers.LeakyReLU(alpha=0.2))


def mlp_3(model_name, k, dataset, seed, activation_func):
    ds_shape, nuser, nitems, ds_code = dataset.get_shape(), dataset.get_num_users(), dataset.get_num_users(), dataset.get_data_code()
    # Do not put a dataset function call in lambda inside a funciton(mlp_2), keras get a reference and try to serialize it when the model is saved.
    # Very weird situation, dificult to debug. It works in the main file but not when put inside a function
    
    input_layer = layers.Input(shape=ds_shape, name="entrada")

    e_user = layers.Dense(
        k,
        name="emb_u",
        use_bias=False,
    )(layers.Lambda(lambda x: x[:, 0:nuser])(input_layer))
    
    e_item = layers.Dense(
        k,
        name="emb_i",
        use_bias=False,
    )(layers.Lambda(lambda x: x[:, nitems:])(input_layer))
    
    x_u = layers.Dense(128, activation=activation_func)(e_user)
    x_u = layers.Dropout(0.1)(x_u)
    x_u = layers.Dense(64, activation=activation_func)(x_u)
    x_u = layers.Dropout(0.1)(x_u)
    x_u = layers.Dense(32, activation=activation_func)(x_u)
    
    x_i = layers.Dense(128, activation=activation_func)(e_item)
    x_i = layers.Dropout(0.1)(x_i)
    x_i = layers.Dense(64, activation=activation_func)(x_i)
    x_i = layers.Dropout(0.1)(x_i)
    x_i = layers.Dense(32, activation=activation_func)(x_i)
    
    x_embedding = layers.Concatenate(axis=1)([x_u, x_i])
    x_embedding = layers.Dense(64, activation=activation_func)(x_embedding)
    x_embedding = layers.Dropout(0.1)(x_embedding)
    x_embedding = layers.Dense(32, activation=activation_func)(x_embedding)
    x_embedding = layers.Dense(16, activation=activation_func)(x_embedding)
    x_embedding = layers.Dense(8, activation=activation_func)(x_embedding)
    outputs = layers.Dense(5)(x_embedding)
    model = keras.Model(inputs=input_layer, outputs=outputs, name=f"{model_name}_{k}_{ds_code}_{seed}")
    model.summary()
    
    return model

def mlp_3_relu(model, k, dataset, seed):
    return mlp_3(model, k, dataset, seed, 'relu')

def mlp_3_tanh(model, k, dataset, seed):
    return mlp_3(model, k, dataset, seed, 'tanh')

def mlp_3_linear(model, k, dataset, seed):
    return mlp_3(model, k, dataset, seed, 'linear')

def mlp_3_lrelu(model, k, dataset, seed):
    return mlp_3(model, k, dataset, seed, tf.keras.layers.LeakyReLU(alpha=0.2))


from keras.regularizers import l2
def gmf(model_name, k, dataset, seed):
    ds_shape, nuser, nitems, ds_code = dataset.get_shape(), dataset.get_num_users(), dataset.get_num_users(), dataset.get_data_code()
    # Do not put a dataset function call in lambda inside a funciton(mlp_2), keras get a reference and try to serialize it when the model is saved.
    # Very weird situation, dificult to debug. It works in the main file but not when put inside a function
    regs=[0,0]
    
    input_layer = layers.Input(shape=ds_shape, name="entrada")

    e_user = layers.Dense(
        k,
        name="emb_u",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[0])
    )(layers.Lambda(lambda x: x[:, 0:nuser])(input_layer))
    
    e_item = layers.Dense(
        k,
        name="emb_i",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[1])
    )(layers.Lambda(lambda x: x[:, nitems:])(input_layer))
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(e_user)
    item_latent = Flatten()(e_item)
    
    # Element-wise product of user and item embeddings 
    #predict_vector = merge([user_latent, item_latent], mode = 'mul')
    predict_vector = Multiply()([user_latent, item_latent])
    
    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    #prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(predict_vector)
    # CHANGED to linear for regression
    outputs = Dense(5, activation='linear', kernel_initializer='lecun_uniform', name = 'prediction')(predict_vector)
    
    model = keras.Model(inputs=input_layer, outputs=outputs, name=f"{model_name}_{k}_{ds_code}_{seed}")
    model.summary()
    
    return model


def mlp(model_name, k, dataset, seed):
    ds_shape, nuser, nitems, ds_code = dataset.get_shape(), dataset.get_num_users(), dataset.get_num_users(), dataset.get_data_code()
    # Do not put a dataset function call in lambda inside a funciton(mlp_2), keras get a reference and try to serialize it when the model is saved.
    # Very weird situation, dificult to debug. It works in the main file but not when put inside a function
    regs=[0,0]
    
    input_layer = layers.Input(shape=ds_shape, name="entrada")

    e_user = layers.Dense(
        k,
        name="emb_u",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[0])
    )(layers.Lambda(lambda x: x[:, 0:nuser])(input_layer))
    
    e_item = layers.Dense(
        k,
        name="emb_i",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[1])
    )(layers.Lambda(lambda x: x[:, nitems:])(input_layer))
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(e_user)
    item_latent = Flatten()(e_item)
    
    # Element-wise product of user and item embeddings 
    #predict_vector = merge([user_latent, item_latent], mode = 'mul')
    vector = Concatenate()([user_latent, item_latent])
    # MLP layers
    for idx in [64,32,16,8]:
        layer = Dense(idx, kernel_regularizer= l2(0), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)
    
    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    #prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(predict_vector)
    # CHANGED to linear for regression
    outputs = Dense(5, activation='linear', kernel_initializer='lecun_uniform', name = 'prediction')(vector)
    
    model = keras.Model(inputs=input_layer, outputs=outputs, name=f"{model_name}_{k}_{ds_code}_{seed}")
    model.summary()
    
    return model


def neumf(model_name, k, dataset, seed):
    ds_shape, nuser, nitems, ds_code = dataset.get_shape(), dataset.get_num_users(), dataset.get_num_users(), dataset.get_data_code()
    # Do not put a dataset function call in lambda inside a funciton(mlp_2), keras get a reference and try to serialize it when the model is saved.
    # Very weird situation, dificult to debug. It works in the main file but not when put inside a function
    regs=[0,0]
    
    input_layer = layers.Input(shape=ds_shape, name="entrada")

    # In order to maintain a good comparsion we must use the same k in the model
    k=int(k/2)

    """
        MLP
    """
    MLP_e_user = layers.Dense(
        k,
        name="mlp_emb_u",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[0])
    )(layers.Lambda(lambda x: x[:, 0:nuser])(input_layer))
    
    MLP_e_item = layers.Dense(
        k,
        name="mlp_emb_i",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[1])
    )(layers.Lambda(lambda x: x[:, nitems:])(input_layer))
    
    # Crucial to flatten an embedding vector!
    MLP_user_latent = Flatten()(MLP_e_user)
    MLP_item_latent = Flatten()(MLP_e_item)
    
    # Element-wise product of user and item embeddings 
    #predict_vector = merge([user_latent, item_latent], mode = 'mul')
    vector = Concatenate()([MLP_user_latent, MLP_item_latent])
    # MLP layers
    for idx in [64,32,16,8]:
        layer = Dense(idx, kernel_regularizer= l2(0), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)
    
    MLP_vector = vector
    
    """
        GMF
    """
    MF_e_user = layers.Dense(
        k,
        name="mf_emb_u",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[0])
    )(layers.Lambda(lambda x: x[:, 0:nuser])(input_layer))
    
    MF_e_item = layers.Dense(
        k,
        name="mf_emb_i",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[1])
    )(layers.Lambda(lambda x: x[:, nitems:])(input_layer))
    
    # Crucial to flatten an embedding vector!
    MF_user_latent = Flatten()(MF_e_user)
    MF_item_latent = Flatten()(MF_e_item)
    
    # Element-wise product of user and item embeddings 
    #predict_vector = merge([user_latent, item_latent], mode = 'mul')
    MF_vector = Multiply()([MF_user_latent, MF_item_latent])
    
    
    """
        NeuMF
    """
    vector = Concatenate()([MF_vector, MLP_vector])
    
    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    #prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(predict_vector)
    # CHANGED to linear for regression
    outputs = Dense(5, activation='linear', kernel_initializer='lecun_uniform', name = 'prediction')(vector)
    
    model = keras.Model(inputs=input_layer, outputs=outputs, name=f"{model_name}_{k}_{ds_code}_{seed}")
    model.summary()
    
    return model


def store_model(model, history, result, outdir):
    model.save(outdir+'/' + model.name + '.h5')
    
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    
    plt.savefig(outdir+'/' + model.name + '.png')
    plt.clf()
    
    f = open(outdir+'/' + model.name + '.mae.result', "w")
    f.write(f"{result};Evaluate on test data test loss, test acc")
    f.close()

def get_model(model, k, dataset, seed):
    init_random(seed) # Before build the model, seed is set up
    model = eval(model+"(model, k, dataset, seed)")
    return model

def get_model_list():
    #return ["neumf"]
    #return ["mlp_3_lrelu", "mlp_3_relu", "mlp_3_linear", "mlp_3_tanh", "mlp_2_lrelu", "mlp_2_relu", "mlp_2_linear", "mlp_2_tanh", "gmf", "mlp", "neumf"]
    return ["mlp_3_lrelu"]
