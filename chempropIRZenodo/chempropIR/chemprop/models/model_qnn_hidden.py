from argparse import Namespace
import math

import torch
import torch.nn as nn

from .mpn import MPN, MPNEncoder
from chemprop.nn_utils import get_activation_function, initialize_weights

import pennylane as qml


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

        if hasattr(args,'frzn_mpn_checkpoint'):
            if args.frzn_mpn_checkpoint is not None:
                for param in self.encoder.parameters():
                    param.requires_grad=False

        if not hasattr(args,'mpn_output_only'):
            self.mpn_output_only = False
        else:
            self.mpn_output_only = args.mpn_output_only

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        bias = args.bias_ffn
        self.output_activation = args.output_activation
        self.norm_range = args.norm_range
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        

        # QNN device
        num_qubits = int(math.log2(first_linear_dim))
        dev = qml.device("default.qubit", wires=num_qubits)
        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def qnode(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(num_qubits), normalize=True)
            qml.BasicEntanglerLayers(weights, wires=range(num_qubits))
            return qml.probs(wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
        
        weight_shapes = {"weights": (args.qnn_layer, num_qubits)}
        qnn_layer = qml.qnn.TorchLayer(qnode, weight_shapes)


        # Create FFN layers
        if args.ffn_num_layers == 2:
            ffn = [
                qnn_layer,
                nn.Linear(first_linear_dim, args.output_size, bias=bias)
            ]
        else:
            ffn = [
                qnn_layer,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            if args.ffn_hidden_size > 3:
                for _ in range(args.ffn_num_layers - 3):
                    ffn.extend([
                        activation,
                        dropout,
                        nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size, bias=bias),
                    ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size, bias=bias),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        
        # Bypass training whole model if only returning fingerprint
        if self.mpn_output_only:
            output = self.encoder(*input)
            return output

        output = self.ffn(self.encoder(*input))

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

         # Positive Value Mapping
        if self.output_activation=='exp':
            output = torch.exp(output)
        if self.output_activation=='ReLu':
            output = nn.ReLU(output)

         # Normalization Mapping
        if self.norm_range is not None:
            norm_data=output[:,self.norm_range[0]:self.norm_range[1]]
            norm_sum=torch.sum(norm_data,1)
            norm_sum=torch.unsqueeze(norm_sum,1)
            output = torch.div(output,norm_sum)

        return output


def build_qnn_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass')
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model
