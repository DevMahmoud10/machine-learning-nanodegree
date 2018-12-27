import keras
from keras import backend


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def buildNN(n_users, n_games, n_latent_factors):
    game_input = keras.layers.Input(shape=[1],name='Item')
    game_embedding = keras.layers.Embedding(n_games + 1, n_latent_factors, name='Game-Embedding')(game_input)
    game_vec = keras.layers.Flatten(name='FlattenGames')(game_embedding)

    user_input = keras.layers.Input(shape=[1],name='User')
    user_embedding = keras.layers.Embedding(n_users + 1, n_latent_factors,name='User-Embedding')(user_input)
    user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)

    prod = keras.layers.dot([game_vec, user_vec], axes= 1, name='DotProduct')
    model = keras.Model([user_input, game_input], prod)
    # model.compile('adam', 'mean_squared_error')
    model.compile(loss=rmse, optimizer='adamax')
    return model
