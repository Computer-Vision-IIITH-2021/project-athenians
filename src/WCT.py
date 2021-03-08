import torch
from torch.utils.serialization import load_lua
import torch.nn as nn

from encoder_decoder_1 import decoder1, encoder1
from encoder_decoder_2 import decoder2, encoder2
from encoder_decoder_3 import decoder3, encoder3
from encoder_decoder_4 import decoder4, encoder4
from encoder_decoder_5 import decoder5, encoder5


class WCT(nn.Module):
    """Class to perform whitening and coloring transform on the features"""

    def __init__(self, models):
        """Initialise using trained models"""

        super(WCT, self).__init__()

        # Load encoders and decoders from lua format
        vgg_1 = load_lua(models.vgg1)
        decoder_vgg_1 = load_lua(models.decoder1)

        vgg_2 = load_lua(models.vgg2)
        decoder_vgg_2 = load_lua(models.decoder2)

        vgg_3 = load_lua(models.vgg3)
        decoder_vgg_3 = load_lua(models.decoder3)

        vgg_4 = load_lua(models.vgg4)
        decoder_vgg_4 = load_lua(models.decoder4)

        vgg_5 = load_lua(models.vgg5)
        decoder_vgg_5 = load_lua(models.decoder5)

        # Use models to get encoders and decoders
        self.e1 = encoder1(vgg_1)
        self.d1 = decoder1(decoder_vgg_1)

        self.e2 = encoder2(vgg_2)
        self.d2 = decoder2(decoder_vgg_2)

        self.e3 = encoder3(vgg_3)
        self.d3 = decoder3(decoder_vgg_3)

        self.e4 = encoder4(vgg_4)
        self.d4 = decoder4(decoder_vgg_4)

        self.e5 = encoder5(vgg_5)
        self.d5 = decoder5(decoder_vgg_5)

    def whiten_and_color(
        self,
        content_features,
        style_features,
        eigenvalue_thresh=1e-5,
        content_reg_weight=1,
        style_reg_weight=0,
    ):
        """Perform whitening and coloring transform using the content and style features"""

        # Mean center the content features
        content_mean = torch.mean(content_features, 1)
        content_mean = content_mean.unsqueeze(1).expand_as(content_features)
        content_features = content_features - content_mean

        content_features_size = content_features.size()

        # Compute covariance matrix for the content with regularisation
        content_covariance = (
            (content_features @ content_features.t()) / (content_features_size[1] - 1)
        ) + (
            content_reg_weight * torch.eye(content_features_size[0], dtype=torch.double)
        )

        # Perform SVD on the content covariance matrix
        content_u, content_s, content_v = torch.svd(content_covariance, some=False)

        # Iterate over the eigenvalues to find index of eigenvectors with eigenvalue more than thresh
        content_eigenvalue_index = content_features_size[0]
        for i in range(content_features_size[0]):
            if content_s[i] < eigenvalue_thresh:
                content_eigenvalue_index = i
                break

        # Perform whitening transform on the content using the formula given in the paper
        content_d_minus_half = content_s[0:content_eigenvalue_index] ** (-0.5)
        content_d_minus_half = torch.diag(content_d_minus_half)

        content_e = content_v[:, 0:content_eigenvalue_index]
        content_e_t = content_v[:, 0:content_eigenvalue_index].t()

        whitened_content = (
            content_e @ content_d_minus_half @ content_e_t @ content_features
        )

        # Mean center style features
        style_mean = torch.mean(style_features, 1)
        style_mean_copy = style_mean.clone().detach()
        style_mean = style_mean.unsqueeze(1).expand_as(style_features)
        style_features = style_features - style_mean

        style_features_size = style_features.size()

        # Compute covariance matrix for the style with regularisation
        style_covariance = (
            (style_features @ style_features.t()) / (style_features_size[1] - 1)
        ) + (style_reg_weight * torch.eye(style_features_size[0], dtype=torch.double))

        # Perform SVD on the style covariance matrix
        style_u, style_s, style_v = torch.svd(style_covariance, some=False)

        # Iterate over the eigenvalues to find index of eigenvectors with eigenvalue more than thresh
        style_eigenvalue_index = style_features_size[0]
        for i in range(style_features_size[0]):
            if style_s[i] < eigenvalue_thresh:
                style_eigenvalue_index = i
                break

        # Perform coloring transform on the whitened content using the formula given in the paper
        style_d_half = style_s[0:style_eigenvalue_index] ** (0.5)
        style_d_half = torch.diag(style_d_half)

        style_e = style_v[:, 0:style_eigenvalue_index]
        style_e_t = style_v[:, 0:style_eigenvalue_index].t()

        content_style_features = style_e @ style_d_half @ style_e_t @ whitened_content

        # Recenter with mean style vector
        transformed_features = content_style_features + style_mean_copy.unsqueeze(
            1
        ).expand_as(content_style_features)

        # Return features
        return transformed_features

    def transform(self, content_features, style_features, F_cs, alpha):
        """Performs WCT and allows user control through alpha"""

        # Cast features to double
        content_features = content_features.double()
        style_features = style_features.double()

        # Reshape content features for whitening and coloring transform
        c = content_features.size(0)
        content_features_view = content_features.view(c, -1)
        style_features_view = style_features.view(c, -1)

        # Transform features
        transformed_features = self.whiten_and_color(
            content_features_view, style_features_view
        )
        transformed_features = transformed_features.view_as(content_features)

        # Blend with content features using weight alpha
        blended_features = (
            alpha * transformed_features + (1.0 - alpha) * content_features
        )
        blended_features = blended_features.float().unsqueeze(0)

        # Copy into F_cs
        F_cs.data.resize_(blended_features.size()).copy_(blended_features)
        return F_cs
