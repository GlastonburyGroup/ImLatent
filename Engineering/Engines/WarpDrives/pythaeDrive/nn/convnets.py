# based on: https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/nn/benchmarks/celeba/convnets.py
"""Proposed Neural nets architectures suited for MNIST"""

from typing import List

import torch
import torch.nn as nn

from pythae.models.nn import BaseDecoder, BaseDiscriminator, BaseEncoder

# uncomment this block if adding it to the repo
# from ....base import BaseAEConfig
# from ....base.base_utils import ModelOutput
# from ...base_architectures import BaseDecoder, BaseEncoder

# remove this block if adding it to the repo
from pythae.models.base import BaseAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder


class Encoder_Conv_AE_MED(BaseEncoder):
    """
    A Convolutional encoder Neural net suited for generic 2D and 3D imaging datasets (mainly, medical images like MRI)-64 and Autoencoder-based models.

    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.celeba import Encoder_Conv_AE_MED
            >>> from pythae.models import AEConfig
            >>> model_config = AEConfig(input_dim=(3, 64, 64), latent_dim=64)
            >>> encoder = Encoder_Conv_AE_MED(model_config)
            >>> encoder
            ... Encoder_Conv_AE_MED(
            ...   (layers): ModuleList(
            ...     (0): Sequential(
            ...       (0): Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (1): Sequential(
            ...       (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (2): Sequential(
            ...       (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...   )
            ...   (embedding): Linear(in_features=16384, out_features=64, bias=True)
            ... )



    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import AE
        >>> model = AE(model_config=model_config, encoder=encoder)
        >>> model.encoder == encoder
        ... True

    .. note::

        Please note that this encoder is only suitable for Autoencoder based models since it only
        outputs the embeddings of the input data under the key `embedding`.

        .. code-block::

            >>> import torch
            >>> input = torch.rand(2, 3, 64, 64)
            >>> out = encoder(input)
            >>> out.embedding.shape
            ... torch.Size([2, 64])

    """

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = args.n_channels #3
        self.dim = args.dim #2
        self.n_features = args.n_features #128

        conv_layer = nn.Conv2d if self.dim == 2 else nn.Conv3d
        bn_layer = nn.BatchNorm2d if self.dim == 2 else nn.BatchNorm3d

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                conv_layer(self.n_channels, self.n_features, 4, 2, padding=1),
                bn_layer(self.n_features),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                conv_layer(self.n_features, self.n_features*2, 4, 2, padding=1), bn_layer(self.n_features*2), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                conv_layer(self.n_features*2, self.n_features*4, 4, 2, padding=1), bn_layer(self.n_features*4), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                conv_layer(self.n_features*4, self.n_features*8, 4, 2, padding=1), bn_layer(self.n_features*8), nn.ReLU()
            )
        )

        self.layers = layers
        self.depth = len(layers)

        if self.dim == 2:
            self.embedding = nn.Linear((self.n_features*8) * (self.input_dim[0]//16) * (self.input_dim[1]//16), args.latent_dim)
        else:
            self.embedding = nn.Linear((self.n_features*8) * (self.input_dim[0]//16) * (self.input_dim[1]//16) * (self.input_dim[2]//16), args.latent_dim)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `embedding_layer_i` where
            i is the layer's level."""
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))

        return output


class Encoder_Conv_VAE_MED(BaseEncoder):
    """
    A Convolutional encoder Neural net suited for generic 2D and 3D imaging datasets (mainly, medical images like MRI)-64 and
    Variational Autoencoder-based models.

    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.celeba import Encoder_Conv_VAE_MED
            >>> from pythae.models import VAEConfig
            >>> model_config = VAEConfig(input_dim=(3, 64, 64), latent_dim=64)
            >>> encoder = Encoder_Conv_VAE_MED(model_config)
            >>> encoder
            ... Encoder_Conv_VAE_MED(
            ...   (layers): ModuleList(
            ...     (0): Sequential(
            ...       (0): Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (1): Sequential(
            ...       (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (2): Sequential(
            ...       (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...   )
            ...   (embedding): Linear(in_features=16384, out_features=64, bias=True)
            ...   (log_var): Linear(in_features=16384, out_features=64, bias=True)
            ... )



    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, encoder=encoder)
        >>> model.encoder == encoder
        ... True


    .. note::

        Please note that this encoder is only suitable for Variational Autoencoder based models
        since it outputs the embeddings and the **log** of the covariance diagonal coefficients
        of the input data under the key `embedding` and `log_covariance`.

        .. code-block::

            >>> import torch
            >>> input = torch.rand(2, 3, 64, 64)
            >>> out = encoder(input)
            >>> out.embedding.shape
            ... torch.Size([2, 64])
            >>> out.log_covariance.shape
            ... torch.Size([2, 64])

    """

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = args.n_channels #3
        self.dim = args.dim #2
        self.n_features = args.n_features #128

        conv_layer = nn.Conv2d if self.dim == 2 else nn.Conv3d
        bn_layer = nn.BatchNorm2d if self.dim == 2 else nn.BatchNorm3d

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                conv_layer(self.n_channels, self.n_features, 4, 2, padding=1),
                bn_layer(self.n_features),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                conv_layer(self.n_features, self.n_features*2, 4, 2, padding=1), bn_layer(self.n_features*2), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                conv_layer(self.n_features*2, self.n_features*4, 4, 2, padding=1), bn_layer(self.n_features*4), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                conv_layer(self.n_features*4, self.n_features*8, 4, 2, padding=1), bn_layer(self.n_features*8), nn.ReLU()
            )
        )

        self.layers = layers
        self.depth = len(layers)

        if self.dim == 2:
            self.embedding = nn.Linear((self.n_features*8) * (self.input_dim[0]//16) * (self.input_dim[1]//16), args.latent_dim)
            self.log_var = nn.Linear((self.n_features*8) * (self.input_dim[0]//16) * (self.input_dim[1]//16), args.latent_dim)
        else:
            self.embedding = nn.Linear((self.n_features*8) * (self.input_dim[0]//16) * (self.input_dim[1]//16) * (self.input_dim[2]//16), args.latent_dim)
            self.log_var = nn.Linear((self.n_features*8) * (self.input_dim[0]//16) * (self.input_dim[1]//16) * (self.input_dim[2]//16), args.latent_dim)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding` and the **log** of the diagonal coefficient of the covariance
            matrices under the key `log_covariance`. Optional: The outputs of the layers specified
            in `output_layer_levels` arguments are available under the keys `embedding_layer_i`
            where i is the layer's level.
        """
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
                output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))

        return output


class Encoder_Conv_SVAE_MED(BaseEncoder):
    """
    A Convolutional encoder Neural net suited for generic 2D and 3D imaging datasets (mainly, medical images like MRI)-64 and Hyperspherical autoencoder
    Variational Autoencoder.

    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.celeba import Encoder_Conv_SVAE_MED
            >>> from pythae.models import SVAEConfig
            >>> model_config = SVAEConfig(input_dim=(3, 64, 64), latent_dim=64)
            >>> encoder = Encoder_Conv_SVAE_MED(model_config)
            >>> encoder
            ... Encoder_Conv_SVAE_MED(
            ...   (layers): ModuleList(
            ...     (0): Sequential(
            ...       (0): Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (1): Sequential(
            ...       (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (2): Sequential(
            ...       (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...   )
            ...   (embedding): Linear(in_features=16384, out_features=64, bias=True)
            ...   (log_concentration): Linear(in_features=16384, out_features=1, bias=True)
            ... )



    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import SVAE
        >>> model = SVAE(model_config=model_config, encoder=encoder)
        >>> model.encoder == encoder
        ... True


    .. note::

        Please note that this encoder is only suitable for Hyperspherical Variational Autoencoder
        models since it outputs the embeddings and the **log** of the concentration in the
        Von Mises Fisher distributions under the key `embedding` and `log_concentration`.

        .. code-block::

            >>> import torch
            >>> input = torch.rand(2, 3, 64, 64)
            >>> out = encoder(input)
            >>> out.embedding.shape
            ... torch.Size([2, 64])
            >>> out.log_concentration.shape
            ... torch.Size([2, 1])

    """

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = args.n_channels #3
        self.dim = args.dim #2
        self.n_features = args.n_features #128

        conv_layer = nn.Conv2d if self.dim == 2 else nn.Conv3d
        bn_layer = nn.BatchNorm2d if self.dim == 2 else nn.BatchNorm3d

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                conv_layer(self.n_channels, self.n_features, 4, 2, padding=1),
                bn_layer(self.n_features),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                conv_layer(self.n_features, self.n_features*2, 4, 2, padding=1), bn_layer(self.n_features*2), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                conv_layer(self.n_features*2, self.n_features*4, 4, 2, padding=1), bn_layer(self.n_features*4), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                conv_layer(self.n_features*4, self.n_features*8, 4, 2, padding=1), bn_layer(self.n_features*8), nn.ReLU()
            )
        )

        self.layers = layers
        self.depth = len(layers)

        if self.dim == 2:
            self.embedding = nn.Linear((self.n_features*8) * (self.input_dim[0]//16) * (self.input_dim[1]//16), args.latent_dim)
            self.log_concentration = nn.Linear((self.n_features*8) * (self.input_dim[0]//16) * (self.input_dim[1]//16), 1)
        else:
            self.embedding = nn.Linear((self.n_features*8) * (self.input_dim[0]//16) * (self.input_dim[1]//16) * (self.input_dim[2]//16), args.latent_dim)
            self.log_concentration = nn.Linear((self.n_features*8) * (self.input_dim[0]//16) * (self.input_dim[1]//16) * (self.input_dim[2]//16), 1)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding` and the **log** of the diagonal coefficient of the covariance
            matrices under the key `log_covariance`. Optional: The outputs of the layers specified
            in `output_layer_levels` arguments are available under the keys `embedding_layer_i`
            where i is the layer's level.
        """
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
                output["log_concentration"] = self.log_concentration(
                    out.reshape(x.shape[0], -1)
                )

        return output


class Decoder_Conv_AE_MED(BaseDecoder):
    """
    A Convolutional decoder Neural net suited for generic 2D and 3D imaging datasets (mainly, medical images like MRI)-64 and Autoencoder-based
    models.

    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.celeba import Decoder_Conv_AE_MED
            >>> from pythae.models import VAEConfig
            >>> model_config = VAEConfig(input_dim=(3, 64, 64), latent_dim=64)
            >>> decoder = Decoder_Conv_AE_MED(model_config)
            >>> decoder
            ... Decoder_Conv_AE_MED(
            ...   (layers): ModuleList(
            ...     (0): Sequential(
            ...       (0): Linear(in_features=64, out_features=65536, bias=True)
            ...     )
            ...     (1): Sequential(
            ...       (0): ConvTranspose2d(1024, 512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (2): Sequential(
            ...       (0): ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
            ...       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (4): Sequential(
            ...       (0): ConvTranspose2d(128, 3, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
            ...       (1): Sigmoid()
            ...     )
            ...   )
            ... )


    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, decoder=decoder)
        >>> model.decoder == decoder
        ... True

    .. note::

        Please note that this decoder is suitable for **all** models.

        .. code-block::

            >>> import torch
            >>> input = torch.randn(2, 64)
            >>> out = decoder(input)
            >>> out.reconstruction.shape
            ... torch.Size([2, 3, 64, 64])
    """

    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = args.n_channels #3
        self.dim = args.dim #2
        self.n_encoder_features = args.n_features #128

        convtrans_layer = nn.ConvTranspose2d if self.dim == 2 else nn.ConvTranspose3d
        bn_layer = nn.BatchNorm2d if self.dim == 2 else nn.BatchNorm3d

        layers = nn.ModuleList()

        if self.dim == 2:
            layers.append(nn.Sequential(nn.Linear(args.latent_dim, (self.n_encoder_features*8) * (self.input_dim[0]//8) * (self.input_dim[1]//8))))
        else:
            layers.append(nn.Sequential(nn.Linear(args.latent_dim, (self.n_encoder_features*8) * (self.input_dim[0]//8) * (self.input_dim[1]//8) * (self.input_dim[2]//8))))

        layers.append(
            nn.Sequential(
                convtrans_layer(self.n_encoder_features*8, self.n_encoder_features*4, 5, 2, padding=2),
                bn_layer(self.n_encoder_features*4),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                convtrans_layer(self.n_encoder_features*4, self.n_encoder_features*2, 5, 2, padding=1, output_padding=0),
                bn_layer(self.n_encoder_features*2),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                convtrans_layer(self.n_encoder_features*2, self.n_encoder_features, 5, 2, padding=2, output_padding=1),
                bn_layer(self.n_encoder_features),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                convtrans_layer(self.n_encoder_features, self.n_channels, 5, 1, padding=1), nn.Sigmoid()
            )
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the reconstruction of the latent code
            under the key `reconstruction`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `reconstruction_layer_i`
            where i is the layer's level.
        """
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z

        for i in range(max_depth):
            out = self.layers[i](out)

            if i == 0:
                if self.dim == 2:
                    out = out.reshape(z.shape[0], self.n_encoder_features*8, (self.input_dim[0]//8), (self.input_dim[1]//8))
                else:
                    out = out.reshape(z.shape[0], self.n_encoder_features*8, (self.input_dim[0]//8), (self.input_dim[1]//8), (self.input_dim[2]//8))

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output


class Discriminator_Conv_MED(BaseDiscriminator):
    """
    A Convolutional discriminator Neural net suited for generic 2D and 3D imaging datasets (mainly, medical images like MRI).


    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.celeba import Discriminator_Conv_MED
            >>> from pythae.models import VAEGANConfig
            >>> model_config = VAEGANConfig(input_dim=(3, 64, 64), latent_dim=64)
            >>> discriminator = Discriminator_Conv_MED(model_config)
            >>> discriminator
            ... Discriminator_Conv_MED(
            ...   (layers): ModuleList(
            ...     (0): Sequential(
            ...       (0): Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (1): Sequential(
            ...       (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): Tanh()
            ...     )
            ...     (2): Sequential(
            ...       (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (4): Sequential(
            ...       (0): Linear(in_features=16384, out_features=1, bias=True)
            ...       (1): Sigmoid()
            ...     )
            ...   )
            ... )


    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAEGAN
        >>> model = VAEGAN(model_config=model_config, discriminator=discriminator)
        >>> model.discriminator == discriminator
        ... True
    """

    def __init__(self, args: dict):
        BaseDiscriminator.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = args.n_channels
        self.dim = args.dim #2
        self.n_features = args.n_features # 128

        self.discriminator_input_dim = args.discriminator_input_dim

        conv_layer = nn.Conv2d if self.dim == 2 else nn.Conv3d
        bn_layer = nn.BatchNorm2d if self.dim == 2 else nn.BatchNorm3d

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                conv_layer(self.n_channels, self.n_features, 4, 2, padding=1),
                bn_layer(self.n_features),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                conv_layer(self.n_features, self.n_features*2, 4, 2, padding=1), bn_layer(self.n_features*2), nn.Tanh()
            )
        )

        layers.append(
            nn.Sequential(
                conv_layer(self.n_features*2, self.n_features*4, 4, 2, padding=1), bn_layer(self.n_features*4), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                conv_layer(self.n_features*4, self.n_features*8, 4, 2, padding=1), bn_layer(self.n_features*8), nn.ReLU()
            )
        )

        if self.dim == 2:
            layers.append(nn.Sequential(nn.Linear((self.n_features*8) * (self.input_dim[0]//16) * (self.input_dim[1]//16), 1), nn.Sigmoid()))
        else:    
            layers.append(nn.Sequential(nn.Linear((self.n_features*8) * (self.input_dim[0]//16) * (self.input_dim[1]//16) * (self.input_dim[2]//16), 1), nn.Sigmoid()))

        self.layers = layers
        self.depth = len(layers)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the adversarial score of the input
            under the key `embedding`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `embedding_layer_i` where
            i is the layer's level.
        """

        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x

        for i in range(max_depth):

            if i == 4:
                out = out.reshape(x.shape[0], -1)

            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = out

        return output