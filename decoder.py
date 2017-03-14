from layers import *


class Decoder(object):
    def __init__(self, n_latent, n_units, n_output, activation):
        self._n_latent = n_latent
        self._n_units = n_units
        self._n_output = n_output
        self._activation = activation
        self._weight_decay_loss = 0.0

        self._n_layer = len(n_units)
        self._hidden_layers = list()
        for layer_idx in range(self._n_layer):
            layer_name = "decoder_layer_" + str(layer_idx + 1)
            if layer_idx is 0:
                n_layer_input = n_latent
            else:
                n_layer_input = n_units[layer_idx - 1]
            n_unit = n_units[layer_idx]

            self._hidden_layers.append(
                AffinePlusNonlinearLayer(layer_name, n_layer_input, n_unit, activation))

        layer_name = "reconstruct_layer"
        self._reconstruct_layer = AffinePlusNonlinearLayer(layer_name, n_units[-1], n_output, activation)

        for layer_idx in range(self._n_layer):
            self._weight_decay_loss += self._hidden_layers[layer_idx].get_weight_decay_loss()
        self._weight_decay_loss += self._reconstruct_layer.get_weight_decay_loss()

    def forward(self, input_tensor):
        output_tensor = input_tensor
        for layer_idx in range(self._n_layer):
            output_tensor = self._hidden_layers[layer_idx].forward(output_tensor)
        output_tensor = self._reconstruct_layer.forward(output_tensor)

        return output_tensor

    @property
    def wd_loss(self):
        return self._weight_decay_loss
