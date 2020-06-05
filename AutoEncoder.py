from keras.layers import Input, Dense
from keras.models import Model


class AutoEncoder:

    def __init__(self, data, test_data, data_name, layers, encoding_dim,
                 batch_size, epochs, rho, lbda, gamma, optimizer, activation):
        """
        Constructs an autoencoder of deep autoencoder.
        :param data: The dataset we want the model to fit to.
        :param data_name: Dataset name in order to save and load weights.
        :param layers: List that defines the topology of the model.
        :param encoding_dim: The bottleneck of the DAE.
        :param batch_size: The number of examples the DAE uses to learn in one epoch.
        :param epochs: Number of learning cycles for the DAE.
        :param rho: The penalty parameter (rho > 0) which control how close Y and the hidden features are.
        :param lbda: The lambda parameter which defines the trade-off between the network objective and the clustering objective.
        :param gamma: The gamma parameter which defines the constructive inductions influence.
        :param optimizer: The DAEs optimizer.
        :param activation: The activation function neurons used to map the result to a real number.
        :return:
        """
        self.data = data
        self.data_name = data_name
        self.layers = layers
        self.encoding_dim = encoding_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.io_shape = (self.layers[0],)
        self.encoding_idx = layers.index(encoding_dim)
        self.rho = rho
        self.lbda = lbda
        self.gamma = gamma
        self.construct_quality = 1.0
        self.optimizer = optimizer
        self.activation = activation

    def _encoder(self):
        """
        Makes encoder based on topology defined in constructor.
        """

        inputs = Input(shape=self.io_shape)
        encoded = Dense(self.layers[1], activation=self.activation)(inputs)

        for l in self.layers[2:self.encoding_idx + 1]:
            encoded = Dense(l, activation=self.activation)(encoded)

        model = Model(inputs, encoded)
        self.encoder = model
        return model

    def _decoder(self):
        """
        Makes decoder based on topology defined in constructor.
        """

        inputs = Input(shape=(self.encoding_dim,))

        decoded = Dense(self.layers[self.encoding_idx + 1])(inputs)

        for l in self.layers[(self.encoding_idx + 2):]:
            decoded = Dense(l, activation=self.activation)(decoded)

        model = Model(inputs, decoded)
        self.decoder = model
        return model

    def encoder_decoder(self):
        """
        Combines encoder and decoder into one model.
        """

        ec = self._encoder()
        dc = self._decoder()

        inputs = Input(self.io_shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)

        self.model = model
        return model

    def fit(self):
        """
        Compile model and fit it to the given data.
        """

        self.model.compile(optimizer=self.optimizer, loss='mse', metrics=['mae'])
        self.model.fit(self.data, self.data, epochs=self.epochs, batch_size=self.batch_size, shuffle=True)

    def print_weights_and_biases(self):
        """
        Prints all weights and biases.
        """

        i = 0

        print('-----------------------')

        for l in self.model.layers:

            print('Id: ', i)

            list = l.get_weights()

            if len(list) != 0:
                print('Weights:')
                print(list[0])
                print('Biases:')
                print(list[1])

            i = i + 1
            print('-----------------------')
