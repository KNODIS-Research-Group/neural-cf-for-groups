import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import pandas as pd
import csv

"""
model, getting index by name. HARD-HARD CODE
coupled and related to ncf_models.py
"""
def get_embeddings(model):
    ## Virtual User
    if "evodeep" in model.name or "mlp" in model.name:
        print("Getting results user from:" + model.layers[3].name)
        userembeddinglayer = model.layers[3]
        print("Getting results items from:" + model.layers[3].name)
        itemembeddinglayer = model.layers[4]
        return (userembeddinglayer, itemembeddinglayer, None, None) # For NeuMF
    else:
        raise ValueError("User embedding not identified Model ")


def extract_embeddings(model, onehot, embedding):
    model_embeddings = keras.Model(inputs = model.input, outputs = embedding.output)
    return model_embeddings.predict(onehot)

def generate_onehot_all_users(dataset):
    onehot_all_user = np.zeros((dataset.get_num_users(), dataset.get_num_users()+dataset.get_num_items()), dtype=np.int32)
    for user_ix in range(dataset.get_num_users()):
        onehot_all_user[user_ix][user_ix] = 1
    return onehot_all_user

def generate_onehot_all_items(dataset):
    onehot_all_item = np.zeros((dataset.get_num_items(), dataset.get_num_users()+dataset.get_num_items()), dtype=np.int32)
    for item_ix in range(dataset.get_num_items()):
        onehot_all_item[item_ix][dataset.get_num_users()+item_ix] = 1
    return onehot_all_item

"""
return (user_embedings, items_embeding)
"""
def extract_embeddings_outputs(model, userembeddinglayer, itemembeddinglayer, dataset):
    return (
                extract_embeddings(model, generate_onehot_all_users(dataset), userembeddinglayer),
                extract_embeddings(model, generate_onehot_all_items(dataset), itemembeddinglayer)
    )


def adapt_evo(model, userembeddinglayer, itemembeddinglayer):
    in_u = layers.Input(shape=(userembeddinglayer.units), name="in_u")
    in_i = layers.Input(shape=(itemembeddinglayer.units), name="in_i")

    #x_u = layers.Dropout(0.1)(e_user)
    #x_i = layers.Dropout(0.1)(e_item)
    x_u = model.layers[5](in_u)
    x_i = model.layers[6](in_i)

    #x_u = layers.Dense(44, activation='relu', kernel_initializer='he_uniform')(x_u)
    #x_i = layers.Dense(44, activation='relu', kernel_initializer='he_uniform')(x_i)
    x_u = model.layers[7](x_u)
    x_i = model.layers[8](x_i)

    #x_d = Multiply()([x_u, x_i])
    x_d = model.layers[9]([x_u, x_i])

    #x_d = layers.Dense(24, activation='relu', kernel_initializer='he_normal')(x_d)
    x_d = model.layers[10](x_d)
    #x_d = layers.Dense(16, activation='elu', kernel_initializer='zero')(x_d)
    x_d = model.layers[11](x_d)

    #outputs = Dense(1, activation='linear', kernel_initializer='lecun_uniform', name = 'prediction')(x_d)
    outputs = model.layers[12](x_d)

    #model = keras.Model(inputs=input_layer, outputs=outputs, name=model_name)
    #model.summary()
    virtual_usermodel = keras.Model(inputs = [in_u, in_i], outputs = outputs)
    return virtual_usermodel


def adapt_mlp(model, userembeddinglayer, itemembeddinglayer):
    in_u = layers.Input(shape=(userembeddinglayer.units), name="in_u")
    in_i = layers.Input(shape=(itemembeddinglayer.units), name="in_i")

    # Crucial to flatten an embedding vector!
    #user_latent = Flatten()(e_user)
    #item_latent = Flatten()(e_item)
    x_u = model.layers[5](in_u)
    x_i = model.layers[6](in_i)
    
    # Element-wise product of user and item embeddings 
    #predict_vector = merge([user_latent, item_latent], mode = 'mul')
    #vector = Concatenate()([user_latent, item_latent])
    x_d = model.layers[7]([x_u, x_i])
    
    # MLP layers
    #for idx in [64,32,16,8]:
    #    layer = Dense(idx, kernel_regularizer= l2(0), activation='relu', name = 'layer%d' %idx)
    #    vector = layer(vector)
    x_d = model.layers[8](x_d)
    x_d = model.layers[9](x_d)
    x_d = model.layers[10](x_d)
    x_d = model.layers[11](x_d)
    
    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    #prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(predict_vector)
    # CHANGED to linear for regression
    #outputs = Dense(1, activation='linear', kernel_initializer='lecun_uniform', name = 'prediction')(vector)
    outputs = model.layers[12](x_d)
    
    #model = keras.Model(inputs=input_layer, outputs=outputs, name=model_name)
    
    virtual_usermodel = keras.Model(inputs = [in_u, in_i], outputs = outputs)
    return virtual_usermodel


def get_fromembeddings_model(model, userembeddinglayer, itemembeddinglayer, userembeddinglayer2, itemembeddinglayer2):
    if "evodeep" in model.name:
        return adapt_evo(model, userembeddinglayer, itemembeddinglayer)
    if "mlp" in model.name:
        return adapt_mlp(model, userembeddinglayer, itemembeddinglayer)
    else:
        raise ValueError("User embedding not identified Model ")