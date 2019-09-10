import torch.nn.functional as F
from torch.autograd import Variable

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from optimizers.darts.genotypes import PRIMITIVES
from optimizers.darts.operations import *


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            '''
            Not used in NASBench
            if 'pool' in primitive:
              op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            '''
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class ChoiceBlock(nn.Module):
    """
    Adapted to match Figure 3 in:
    Bender, Gabriel, et al. "Understanding and simplifying one-shot architecture search."
    International Conference on Machine Learning. 2018.
    """

    def __init__(self, C_in, n_inputs):
        super(ChoiceBlock, self).__init__()
        # Pre-processing 1x1 convolution at the beginning of each choice block.
        self.preprocess = ConvBnRelu(C_in=C_in * n_inputs, C_out=C_in, kernel_size=1, stride=1, padding=0)
        self.mixed_op = MixedOp(C_in, stride=1)

    def forward(self, inputs, input_weights, weights):
        if input_weights is not None:
            # Weigh the input to the choice block
            inputs = [w * t for w, t in zip(input_weights[0], inputs)]

        # Concatenate input to choice block and apply 1x1 convolution
        pre_inputs = self.preprocess(torch.cat(inputs, dim=1))
        # Apply Mixed Op
        output = self.mixed_op(pre_inputs, weights=weights)
        return output


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev, C, layer, search_space):
        super(Cell, self).__init__()
        # All cells are normal cells in NASBench case.
        if layer == 0:
            C_in = C_prev
        else:
            C_in = C_prev * multiplier

        # For preprocessing use the same convbnrelu function, which is what NASBench is using
        self.preprocess = ConvBnRelu(C_in=C_in, C_out=C, kernel_size=1, stride=1, padding=0)

        self._steps = steps
        self._multiplier = multiplier

        self._choice_blocks = nn.ModuleList()
        self._bns = nn.ModuleList()
        self.search_space = search_space

        # Create the choice block.
        for i in range(self._steps):
            choice_block = ChoiceBlock(C_in=C, n_inputs=i + 1)
            self._choice_blocks.append(choice_block)

    def forward(self, s0, weights, output_weights, input_weights):
        # Adaption to NASBench
        # Only use a single input, from the previous cell
        s0 = self.preprocess(s0)

        states = [s0]
        # Loop that connects all previous cells to the current one.
        for i in range(self._steps):
            # Select the current weighting for input edges to each choice block
            if input_weights is not None:
                # Node 1 has no choice with respect to its input
                if (i == 0) or (i == 1 and type(self.search_space) == SearchSpace1):
                    input_weight = None
                else:
                    input_weight = input_weights.pop(0)

            # Iterate over the choice blocks
            s = self._choice_blocks[i](inputs=states, input_weights=input_weight, weights=weights[i])
            states.append(s)
        assert (len(input_weights) == 0, 'Something went wrong here.')
        # Concatenate all mixed op outputs, the slicing ignores the input op.
        # Create weighted concatenation at the output of the cell
        if output_weights is None:
            weighted_tensor_list = states[-self._multiplier:]
        else:
            weighted_tensor_list = [w * t for w, t in zip(output_weights[0], states[-self._multiplier:])]
        return torch.cat(weighted_tensor_list, dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, output_weights, search_space, steps=4, multiplier=5):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._output_weights = output_weights
        self.nasbench = None
        self.search_space = search_space

        # In NASBench the stem has 128 output channels
        C_curr = C
        self.stem = ConvBnRelu(C_in=3, C_out=C_curr, kernel_size=3, stride=1)

        self.cells = nn.ModuleList()
        C_prev = C_curr
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                # Double the number of channels after each down-sampling step
                # Down-sample in forward method
                C_curr *= 2

            cell = Cell(steps, multiplier, C_prev, C_curr, layer=i, search_space=search_space)
            self.cells += [cell]
            C_prev = C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.postprocess = ReLUConvBN(C_in=C_prev * multiplier, C_out=C_curr, kernel_size=1, stride=1, padding=0,
                                      affine=False)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion,
                            steps=self.search_space.num_intermediate_nodes, output_weights=self._output_weights,
                            search_space=self.search_space).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _preprocess_op(self, x, discrete, normalize):
        if discrete and normalize:
            raise ValueError("architecture can't be discrete and normalized")
        # If using discrete architecture from random_ws search with weight sharing then pass through architecture
        # weights directly.
        if discrete:
            return x
        elif normalize:
            arch_sum = torch.sum(x, dim=-1)
            if arch_sum > 0:
                return x / arch_sum
            else:
                return x
        else:
            # Normal search softmax over the inputs and mixed ops.
            return F.softmax(x, dim=-1)

    def forward(self, input, discrete=False, normalize=False):
        # NASBench only has one input to each cell
        s0 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if i in [self._layers // 3, 2 * self._layers // 3]:
                # Perform down-sampling by factor 1/2
                # Equivalent to https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L68
                s0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)(s0)

            # Normalize mixed_op weights for the choice blocks in the graph
            mixed_op_weights = self._preprocess_op(self._arch_parameters[0], discrete=discrete, normalize=False)

            # Normalize the output weights
            output_weights = self._preprocess_op(self._arch_parameters[1], discrete,
                                                 normalize) if self._output_weights else None
            # Normalize the input weights for the nodes in the cell
            input_weights = [self._preprocess_op(alpha, discrete, normalize) for alpha in self._arch_parameters[2:]]
            s0 = cell(s0, mixed_op_weights, output_weights, input_weights)

        # Include one more preprocessing step here
        s0 = self.postprocess(s0)  # [N, C_max * multiplier, w, h] -> [N, C_max, w, h]

        # Global Average Pooling by averaging over last two remaining spatial dimensions
        # Like in nasbench: https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L92
        out = s0.view(*s0.shape[:2], -1).mean(-1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        # Initializes the weights for the mixed ops.
        num_ops = len(PRIMITIVES)
        self.alphas_mixed_op = Variable(1e-3 * torch.randn(self._steps, num_ops).cuda(), requires_grad=True)

        # For the alphas on the output node initialize a weighting vector for all nodes in each layer.
        self.alphas_output = Variable(1e-3 * torch.randn(1, self._steps + 1).cuda(), requires_grad=True)

        # Search space 1 has no weight on node 2 therefore begin at node 3
        if type(self.search_space) == SearchSpace1:
            begin = 3
        else:
            begin = 2
        # Initialize the weights for the inputs to each choice block.
        self.alphas_inputs = [Variable(1e-3 * torch.randn(1, n_inputs).cuda(), requires_grad=True) for n_inputs in
                              range(begin, self._steps + 1)]

        # Total architecture parameters
        self._arch_parameters = [
            self.alphas_mixed_op,
            self.alphas_output,
            *self.alphas_inputs
        ]

    def arch_parameters(self):
        return self._arch_parameters
