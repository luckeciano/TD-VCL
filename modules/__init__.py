from .multi_head_mlp import MultiHeadMLP
from .multi_head_cnn import MultiHeadMNISTCNN
from .conv_net import ConvNet, AlexNet, AlexNetV2, AlexNet64

from .mean_field_gaussian import MeanFieldGaussian, MeanFieldGaussianWithNodeVariance
from .vcl_bayesian_linear import VCLBayesianLinear
from .vcl_bayesian_conv import VCLBayesianConv2D
from .vcl_nstepkl_bayesian_conv import NStepKLVCLBayesianConv2D
from .td_vcl_bayesian_conv import TDVCLBayesianConv2D

from .vcl_nstepkl_conv_net import NStepKLVCLBayesianAlexNet, NStepKLVCLBayesianAlexNetV2, MultiHeadNStepKLVCLBayesianAlexNet
from .td_vcl_conv_net import TDVCLBayesianAlexNet, TDVCLBayesianAlexNetV2, MultiHeadTDVCLBayesianAlexNet

from .vcl_conv_net import VCLBayesianConvNet, VCLBayesianAlexNet, VCLBayesianAlexNetV2, MultiHeadVCLBayesianAlexNet, VCLBayesianAlexNet64, MultiHeadVCLBayesianAlexNet64
from .nstepkl_bayesian_linear import NStepKLVCLBayesianLinear
from .td_bayesian_linear import TDBayesianLinear
from .vcl import VCL
from .vcl_nstepkl import NStepKLVCL
from .td_vcl import TemporalDifferenceVCL

from .ucl_bayesian_linear import UCLBayesianLinear
from .ucl_conv_net import UCLBayesianAlexNet, UCLBayesianAlexNet64
from .ucl import UCL

from .td_ucl_bayesian_linear import TDUCLBayesianLinear
from .td_ucl import TemporalDifferenceUCL
from .td_ucl_conv_net import TDUCLBayesianAlexNet, MultiHeadTDUCLBayesianAlexNet
