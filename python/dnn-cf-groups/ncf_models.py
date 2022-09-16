from data_utils import init_random

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Embedding, Input, Dense, Reshape, Flatten, Concatenate, Multiply
import numpy as np
import matplotlib.pyplot as plt


from keras.regularizers import l2
def gmf(model_name, k, dataset, seed, embedding_activation = None):
    ds_shape, nuser, nitems, ds_code = dataset.get_shape(), dataset.get_num_users(), dataset.get_num_users(), dataset.get_data_code()
    # Do not put a dataset function call in lambda inside a funciton(mlp_2), keras get a reference and try to serialize it when the model is saved.
    # Very weird situation, dificult to debug. It works in the main file but not when put inside a function
    regs=[0,0]

    input_layer = layers.Input(shape=ds_shape, name="entrada")

    e_user = layers.Dense(
        k,
        name="emb_u",
        activation = embedding_activation,
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[0])
    )(layers.Lambda(lambda x: x[:, 0:nuser])(input_layer))

    e_item = layers.Dense(
        k,
        name="emb_i",
        activation = embedding_activation,
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[1])
    )(layers.Lambda(lambda x: x[:, nuser:])(input_layer))

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
    outputs = Dense(1, activation='linear', kernel_initializer='lecun_uniform', name = 'prediction')(predict_vector)

    model = keras.Model(inputs=input_layer, outputs=outputs, name=model_name)
    model.summary()

    return model


def mlp(model_name, k, dataset, seed, embedding_activation = None):
    ds_shape, nuser, nitems, ds_code = dataset.get_shape(), dataset.get_num_users(), dataset.get_num_users(), dataset.get_data_code()
    # Do not put a dataset function call in lambda inside a funciton(mlp_2), keras get a reference and try to serialize it when the model is saved.
    # Very weird situation, dificult to debug. It works in the main file but not when put inside a function
    regs=[0,0]

    input_layer = layers.Input(shape=ds_shape, name="entrada")

    e_user = layers.Dense(
        k,
        name="emb_u",
        activation = embedding_activation,
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[0])
    )(layers.Lambda(lambda x: x[:, 0:nuser])(input_layer))

    e_item = layers.Dense(
        k,
        name="emb_i",
        activation = embedding_activation,
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[1])
    )(layers.Lambda(lambda x: x[:, nuser:])(input_layer))

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
    outputs = Dense(1, activation='linear', kernel_initializer='lecun_uniform', name = 'prediction')(vector)

    model = keras.Model(inputs=input_layer, outputs=outputs, name=model_name)
    model.summary()

    return model


def neumf(model_name, k, dataset, seed, embedding_activation = None):
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
        activation = embedding_activation,
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[0])
    )(layers.Lambda(lambda x: x[:, 0:nuser])(input_layer))

    MLP_e_item = layers.Dense(
        k,
        name="mlp_emb_i",
        activation = embedding_activation,
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[1])
    )(layers.Lambda(lambda x: x[:, nuser:])(input_layer))

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
        activation = embedding_activation,
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[0])
    )(layers.Lambda(lambda x: x[:, 0:nuser])(input_layer))

    MF_e_item = layers.Dense(
        k,
        name="mf_emb_i",
        activation = embedding_activation,
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[1])
    )(layers.Lambda(lambda x: x[:, nuser:])(input_layer))

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
    outputs = Dense(1, activation='linear', kernel_initializer='lecun_uniform', name = 'prediction')(vector)

    model = keras.Model(inputs=input_layer, outputs=outputs, name=model_name)
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

def get_model(model, k, dataset, seed, embedding_activation = None):
    init_random(seed) # Before build the model, seed is set up
    if (embedding_activation!=None):
        modelname=f"{model}_k{k}_ds{dataset.get_data_code()}_seed{seed}_embacti_{embedding_activation}"
    else:
        modelname=f"{model}_k{k}_ds{dataset.get_data_code()}_seed{seed}"
    model = eval(model+"(modelname, k, dataset, seed, embedding_activation)")
    return model

def get_model_list():
    #return ["gmf", "mlp", "neumf"]
    #return ["mlp"]
    #return ["gmf"]
    return ["gmf", "mlp"]
    #return ["mlp_3_lrelu", "mlp_3_relu", "mlp_3_linear", "mlp_3_tanh", "mlp_2_lrelu", "mlp_2_relu", "mlp_2_linear", "mlp_2_tanh", "gmf", "mlp", "neumf"]
