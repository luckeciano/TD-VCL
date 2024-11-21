from .multi_head_mlp import MultiHeadMLP
from .multi_head_cnn import MultiHeadMNISTCNN
from .conv_net import ConvNet, AlexNet, AlexNetV2

from .mean_field_gaussian import MeanFieldGaussian, MeanFieldGaussianWithNodeVariance
from .vcl_bayesian_linear import VCLBayesianLinear
from .vcl_bayesian_conv import VCLBayesianConv2D
from .vcl_nstepkl_bayesian_conv import NStepKLVCLBayesianConv2D
from .td_vcl_bayesian_conv import TDVCLBayesianConv2D

from .vcl_nstepkl_conv_net import NStepKLVCLBayesianAlexNet, NStepKLVCLBayesianAlexNetV2, MultiHeadNStepKLVCLBayesianAlexNetV2
from .td_vcl_conv_net import TDVCLBayesianAlexNet, TDVCLBayesianAlexNetV2, MultiHeadTDVCLBayesianAlexNetV2

from .vcl_conv_net import VCLBayesianConvNet, VCLBayesianAlexNet, VCLBayesianAlexNetV2, MultiHeadVCLBayesianAlexNetV2
from .nstepkl_bayesian_linear import NStepKLVCLBayesianLinear
from .td_bayesian_linear import TDBayesianLinear
from .vcl import VCL
from .vcl_nstepkl import NStepKLVCL
from .td_vcl import TemporalDifferenceVCL

from .ucl_bayesian_linear import UCLBayesianLinear
from .ucl_conv_net import UCLBayesianAlexNet
from .ucl import UCL
