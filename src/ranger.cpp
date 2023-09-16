/*
# Ranger deep learning optimizer - RAdam + Lookahead + Gradient Centralization, combined into one optimizer.

# https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
# and/or
# https://github.com/lessw2020/Best-Deep-Learning-Optimizers

# Ranger has been used to capture 12 records on the FastAI leaderboard.

# This version = 2020.9.4


# Credits:
# Gradient Centralization --> https://arxiv.org/abs/2004.01461v2 (a new optimization technique for DNNs), github:  https://github.com/Yonghongwei/Gradient-Centralization
# RAdam -->  https://github.com/LiyuanLucasLiu/RAdam
# Lookahead --> rewritten by lessw2020, but big thanks to Github @LonePatient and @RWightman for ideas from their code.
# Lookahead paper --> MZhang,G Hinton  https://arxiv.org/abs/1907.08610

# summary of changes:
# 9/4/20 - updated addcmul_ signature to avoid warning.  Integrates latest changes from GC developer (he did the work for this), and verified on performance on private dataset.
# 4/11/20 - add gradient centralization option.  Set new testing benchmark for accuracy with it, toggle with use_gc flag at init.
# full code integration with all updates at param level instead of group, moves slow weights into state dict (from generic weights),
# supports group learning rates (thanks @SHolderbach), fixes sporadic load from saved model issues.
# changes 8/31/19 - fix references to *self*.N_sma_threshold;
# changed eps to 1e-5 as better default than 1e-8.

import math
import torch
from torch.optim.optimizer import Optimizer
#, required

# If dim is None it will be chosen automatically
def centralized_gradient(x, use_gc=True, gc_conv_only=False, dim=None):
    '''credit - https://github.com/Yonghongwei/Gradient-Centralization '''
    if use_gc:
        dim_threshold = 3 if gc_conv_only else 1
        if len(list(x.size())) > dim_threshold:
            x.add_(-x.mean(dim=(dim or tuple(range(1, len(list(x.size()))))), keepdim=True))
    return x


class Ranger(Optimizer):

    def __init__(self, params, lr=1e-3,                       # lr
                 alpha=0.5, k=6, N_sma_threshhold=5,           # Ranger options
                 betas=(.95, 0.999), eps=1e-5, weight_decay=0,  # Adam options
                 # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                 use_gc=True, gc_conv_only=False, gc_loc=True
                 ):

        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                        N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay,
                        gc_dim=None)
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # look ahead params

        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # gc on or off
        self.gc_loc = gc_loc
        self.use_gc = use_gc
        self.gc_conv_only = gc_conv_only
        # level of gradient centralization
        #self.gc_gradient_threshold = 3 if gc_conv_only else 1

        print(
            f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}")
        if (self.use_gc and self.gc_conv_only == False):
            print(f"GC applied to both conv and fc layers")
        elif (self.use_gc and self.gc_conv_only == True):
            print(f"GC applied to conv layers only")

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError(
                        'Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:  # if first time to run...init dictionary with our desired entries
                    # if self.first_run_check==0:
                    # self.first_run_check=1
                    #print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # GC operation for Conv layers and FC layers
                # if grad.dim() > self.gc_gradient_threshold:
                #    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))
                if self.gc_loc:
                    grad = centralized_gradient(grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only, dim=group['gc_dim'])

                state['step'] += 1

                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # compute mean moving avg
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                buffered = self.radam_buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * \
                        state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                            N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                # if group['weight_decay'] != 0:
                #    p_data_fp32.add_(-group['weight_decay']
                #                     * group['lr'], p_data_fp32)

                # apply lr
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    G_grad = exp_avg / denom
                else:
                    G_grad = exp_avg

                if group['weight_decay'] != 0:
                    G_grad.add_(p_data_fp32, alpha=group['weight_decay'])
                # GC operation
                if self.gc_loc == False:
                    G_grad = centralized_gradient(G_grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only, dim=group['gc_dim'])

                p_data_fp32.add_(G_grad, alpha=-step_size * group['lr'])

                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    # get access to slow param tensor
                    slow_p = state['slow_buffer']
                    # (fast weights - slow weights) * alpha
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    # copy interpolated weights to RAdam param tensor
                    p.data.copy_(slow_p)

        return loss

*/
#include <iostream>
#include <fstream>
#include <vector>
#include <initializer_list>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <cmath>
#include <torch/torch.h>

#include "ranger.h"

// If dim is None it will be chosen automatically
torch::Tensor centralized_gradient(torch::Tensor x, bool use_gc = true, bool gc_conv_only = false, int dim = -1)
{

    // credit - https://github.com/Yonghongwei/Gradient-Centralization
    if (use_gc)
    {
        // dim_threshold = 3 if gc_conv_only else 1
        int dim_threshold = 1;
        if (gc_conv_only)
        {
            dim_threshold = 3;
        }
        // if (len(list(x.size())) > dim_threshold) {
        //     x.add_(-x.mean(dim=(dim or tuple(range(1, len(list(x.size()))))), keepdim=True))
        // }
        if (x.sizes().size() > dim_threshold)
        {
            if (dim > 0)
            {
                x.add_(-x.mean(dim, true));
            }
            else
            {
                /*
                std::vector<int64_t> mean_dim;
                for (int64_t i = 1; i < x.sizes().size(); i++) {
                    mean_dim.push_back(i);
                }*/
                // std::initializer_list<int64_t> mean_dim_list(mean_dim);
                // std::optional<torch::ArrayRef<int64_t>> o1(c10::makeArrayRef(mean_dim));
                // c10::OptionalArrayRef<int64_t> oarr(o1);
                int64_t N_dim = x.sizes().size();
                if (N_dim == 2)
                {
                    x.add_(-x.mean({{1}}, true));
                }
                else if (N_dim == 3)
                {
                    x.add_(-x.mean({1, 2}, true));
                }
                else if (N_dim == 4)
                {
                    x.add_(-x.mean({1, 2, 3}, true));
                }
                else
                {
                    throw std::invalid_argument("Tensor dimension is too large, does not support:" + std::to_string(N_dim) + ". Try tensor with dimension <= 4.");
                }
            }
        }
    }
    return x;
}

std::tuple<double, double> compute_buffer(int step, int N_sma_threshhold, double beta1, double beta2)
{
    /*
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * \
                            state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma
                        if N_sma > self.N_sma_threshhold:
                            step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                                N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                        else:
                            step_size = 1.0 / (1 - beta1 ** state['step'])
                        buffered[2] = step_size
    */
    double beta2_t = std::pow(beta2, step);
    double N_sma_max = 2 / (1 - beta2) - 1;
    double N_sma = N_sma_max - 2 * step * beta2_t / (1 - beta2_t);
    // buffered[1] = N_sma

    double step_size = -1;
    if (N_sma > N_sma_threshhold)
    {
        double num = (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2);
        double den = (1 - std::pow(beta1, step));
        if (num >= 0)
        {
            step_size = std::sqrt(num) / den;
        }
        // step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
        //         N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** step)
    }
    else
    {
        step_size = 1.0 / (1 - std::pow(beta1, step));
    }

    // buffered[2] = step_size
    return {N_sma, step_size};
}

torch::Tensor Ranger::step(LossClosure closure)
{
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure != nullptr)
    {
        at::AutoGradMode enable_grad(true);
        loss = closure();
    }

    // Evaluate averages and grad, update param tensors
    for (auto &group : param_groups_)
    {
        for (auto &p : group.params())
        {
            if (!p.grad().defined())
            {
                continue;
            }
            auto grad = p.grad();
            TORCH_CHECK(!grad.is_sparse(), "Adam does not support sparse gradients");

            // auto p_data_fp32 = p.data().float();
            auto p_data_fp32 = p.toType(torch::kFloat32);

            auto param_state = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
            auto &options = static_cast<RangerOptions &>(group.options());

            // std::cout << "Group[" << &p << "]: " << options.lr() << " " << options.gc_dim() << std::endl;

            // State initialization
            if (param_state == state_.end())
            {
                auto state = std::make_unique<RangerParamState>();
                state->step(0);
                // Exponential moving average of gradient values
                state->exp_avg(torch::zeros_like(p, torch::MemoryFormat::Preserve));
                // Exponential moving average of squared gradient values
                state->exp_avg_sq(torch::zeros_like(p, torch::MemoryFormat::Preserve));

                // look ahead weight storage now in state dict
                auto slow_buffer_tensor = torch::empty_like(p, torch::MemoryFormat::Preserve);
                slow_buffer_tensor.copy_(p);
                state->slow_buffer(slow_buffer_tensor);
                // state['slow_buffer'] = torch.empty_like(p.data)
                // state['slow_buffer'].copy_(p.data)

                state_[c10::guts::to_string(p.unsafeGetTensorImpl())] = std::move(state);
            }
            // else
            //{
            //  state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32);
            //  state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32);
            //}

            // begin computations
            auto &state = static_cast<RangerParamState &>(*state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
            auto &exp_avg = state.exp_avg();
            auto &exp_avg_sq = state.exp_avg_sq();

            // reassign type into state
            if (exp_avg.dtype() != p_data_fp32.dtype())
            {
                std::cout << exp_avg.dtype() << "!=" << p_data_fp32.dtype() << std::endl;
                exp_avg.to(p_data_fp32.dtype());
                exp_avg_sq.to(p_data_fp32.dtype());
                state.exp_avg(exp_avg);
                state.exp_avg_sq(exp_avg_sq);
            }

            auto beta1 = std::get<0>(options.betas());
            auto beta2 = std::get<1>(options.betas());

            // GC operation for Conv layers and FC layers
            // if grad.dim() > self.gc_gradient_threshold:
            //    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))
            if (hyperparams.gc_loc())
            {
                // grad = centralized_gradient(grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only, dim=group['gc_dim'])
                grad = centralized_gradient(grad, hyperparams.use_gc(), hyperparams.gc_conv_only(), options.gc_dim());
            }

            state.step(state.step() + 1);

            // compute variance mov avg
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

            // compute mean moving avg
            exp_avg.mul_(beta1).add_(grad, 1 - beta1);

            auto computed_buffered = compute_buffer(state.step(), hyperparams.N_sma_threshhold(), beta1, beta2);
            int N_sma = std::get<0>(computed_buffered);
            double step_size = std::get<1>(computed_buffered);

            // # if group['weight_decay'] != 0:
            // #   p_data_fp32.add_(-group['weight_decay']
            // #                    * group['lr'], p_data_fp32)

            // apply lr
            torch::Tensor G_grad;
            if (N_sma > hyperparams.N_sma_threshhold())
            {
                auto denom = exp_avg_sq.sqrt().add_(options.eps());
                G_grad = exp_avg / denom;
            }
            else
            {
                G_grad = exp_avg;
            }

            // if group['weight_decay'] != 0:
            //     G_grad.add_(p_data_fp32, alpha=group['weight_decay'])

            if (options.weight_decay() != 0)
            {
                G_grad.add(p_data_fp32, options.weight_decay());
            }

            // GC operation
            if (hyperparams.gc_loc() == false)
            {
                G_grad = centralized_gradient(G_grad, hyperparams.use_gc(), hyperparams.gc_conv_only(), options.gc_dim());
            }

            //std::cout << "G_grad = " << G_grad << std::endl;
            p_data_fp32.add_(G_grad, -step_size * options.lr());

            p.copy_(p_data_fp32);

            // integrated look ahead...
            // we do it at the param level instead of group level
            if (state.step() % options.k() == 0)
            {
                // get access to slow param tensor
                auto slow_p = state.slow_buffer();
                // (fast weights - slow weights) * alpha
                slow_p.add_(p - slow_p, hyperparams.alpha());
                // copy interpolated weights to RAdam param tensor
                p.copy_(slow_p);
            }
        }
    }
    return loss;
}

void Ranger::show_hypers()
{
    std::cout << hyperparams.lr() << std::endl;
    std::cout << hyperparams.alpha() << std::endl;
    std::cout << hyperparams.k() << std::endl;
    std::cout << hyperparams.N_sma_threshhold() << std::endl;
    std::cout << "(" << std::get<0>(hyperparams.betas()) << "," << std::get<1>(hyperparams.betas()) << ")" << std::endl;
    std::cout << hyperparams.eps() << std::endl;
    std::cout << hyperparams.weight_decay() << std::endl;
    std::cout << hyperparams.use_gc() << std::endl;
    std::cout << hyperparams.gc_conv_only() << std::endl;
    std::cout << hyperparams.gc_loc() << std::endl;
    std::cout << hyperparams.gc_dim() << std::endl;
}

void Ranger::set_state(torch::Tensor &p, RangerParamState &state)
{
    auto x_state = std::make_unique<RangerParamState>(state);
    state_[c10::guts::to_string(p.unsafeGetTensorImpl())] = std::move(x_state);
}

//// RangerOptions

RangerOptions::RangerOptions(double lr) : lr_(lr) {}

#define copy_prop(name, src) name(src.name())

RangerOptions::RangerOptions(const RangerOptions &other)
{
    copy_prop(lr, other);
    copy_prop(alpha, other);
    copy_prop(k, other);
    copy_prop(N_sma_threshhold, other);
    copy_prop(betas, other);
    copy_prop(eps, other);
    copy_prop(weight_decay, other);
    copy_prop(use_gc, other);
    copy_prop(gc_conv_only, other);
    copy_prop(gc_loc, other);
    copy_prop(gc_dim, other);
}

#undef copy_prop

bool operator==(const RangerOptions &lhs, const RangerOptions &rhs)
{
    std::cout << "Option equal operator is called!!!!!!!!!!!!!!!!" << std::endl;
    return (lhs.lr() == rhs.lr()) &&
           (std::get<0>(lhs.betas()) == std::get<0>(rhs.betas())) &&
           (std::get<1>(lhs.betas()) == std::get<1>(rhs.betas())) &&
           (lhs.eps() == rhs.eps()) &&
           (lhs.weight_decay() == rhs.weight_decay()) &&
           (lhs.alpha() == rhs.alpha()) &&
           (lhs.k() == rhs.k()) &&
           (lhs.N_sma_threshhold() == rhs.N_sma_threshhold()) &&
           (lhs.use_gc() == rhs.use_gc()) &&
           (lhs.gc_conv_only() == rhs.gc_conv_only()) &&
           (lhs.gc_loc() == rhs.gc_loc()) &&
           (lhs.gc_dim() == rhs.gc_dim());
}

void RangerOptions::serialize(torch::serialize::OutputArchive &archive) const
{
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);

    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(alpha);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(k);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(N_sma_threshhold);

    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(betas);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);

    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(use_gc);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(gc_conv_only);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(gc_loc);
}

void RangerOptions::serialize(torch::serialize::InputArchive &archive)
{
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);

    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, alpha);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int, k);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int, N_sma_threshhold);

    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(betas_t, betas);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);

    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, use_gc);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, gc_conv_only);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, gc_loc);
}

double RangerOptions::get_lr() const
{
    return lr();
}

void RangerOptions::set_lr(const double lr)
{
    this->lr(lr);
}

//// RangerParamState

bool operator==(const RangerParamState &lhs, const RangerParamState &rhs)
{
    std::cout << "State equal operator is called!!!!!!!!!!!!!!!!" << std::endl;
    return (lhs.step() == rhs.step()) &&
           torch::equal(lhs.exp_avg(), rhs.exp_avg()) &&
           torch::equal(lhs.exp_avg_sq(), rhs.exp_avg_sq()) &&
           torch::equal_if_defined(lhs.slow_buffer(), rhs.slow_buffer());
}

void RangerParamState::serialize(torch::serialize::OutputArchive &archive) const
{
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
}

void RangerParamState::serialize(torch::serialize::InputArchive &archive)
{
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

torch::Tensor load_txt_tensor(std::ifstream &inf)
{
    int n_dim = 0;
    inf >> n_dim;

    std::vector<int64_t> sizes_vec;
    int64_t total = 1;
    for (int i = 0; i < n_dim; i++)
    {
        int64_t dim_size = 0;
        inf >> dim_size;
        sizes_vec.push_back(dim_size);
        total *= dim_size;
    }
    std::cout << "n_dim = " << n_dim << std::endl;
    std::cout << "(";
    for (int j = 0; j < sizes_vec.size(); j++) {
        if (j > 0) {
            std::cout << " ";
        }
        std::cout << sizes_vec[j];
    }
    std::cout << ")\n";

    float *tensor_value = new float[total];
    for (int i = 0; i < total; i++)
    {
        float value;
        inf >> value;
        // std::cout << value << " " << sizeof(double) << std::endl;
        tensor_value[i] = value;
    }
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    auto result = torch::from_blob(tensor_value, sizes_vec, float_options);
    return result;
}

int load_txt_int(std::ifstream &inf)
{
    int value = 0;
    inf >> value;
    std::cout << "Read int: " << value << std::endl;
    return value;
}

void test_create_ranger_optimizer()
{
    auto model = torch::nn::Linear(10, 1);
    auto optim_option = RangerOptions().lr(0.003).alpha(0.456);
    Ranger optimizer({torch::optim::OptimizerParamGroup(model->parameters())}, optim_option);
    optimizer.show_hypers();
}

void test_centralized_gradient()
{
    std::ifstream inf;
    inf.open("/home/mc/sidework/nnchess/tdlambda-nnue/tdlambda-py/build/grad_central_tests/gc_tests.txt");

    for (int i = 0; i < 3; i++)
    {
        torch::Tensor x = load_txt_tensor(inf);
        torch::Tensor y = load_txt_tensor(inf);
        torch::Tensor yhat = centralized_gradient(x, true, false, 0);

        std::cout << "Case " << i << ":" << std::endl;
        std::cout << y << std::endl;
        std::cout << yhat << std::endl;
        std::cout << "difference norm = " << torch::norm(y - yhat) << std::endl;
    }

    inf.close();
}

void test_compute_buffer()
{
    std::ifstream inf;
    int step, threshold;
    double beta1, beta2, N_sma, step_size;

    inf.open("/home/mc/sidework/nnchess/tdlambda-nnue/tdlambda-py/build/grad_central_tests/buffer_input.txt");
    while (!inf.eof())
    {
        inf >> step >> threshold >> beta1 >> beta2 >> N_sma >> step_size;
        auto ret_pair = compute_buffer(step, threshold, beta1, beta2);
        std::cout << std::get<0>(ret_pair) << " " << N_sma << std::endl;
        std::cout << std::get<1>(ret_pair) << " " << step_size << std::endl;
    }
    inf.close();
}

void test_step_single_param_(torch::Tensor x, torch::Tensor grad,
                             int state_step, torch::Tensor state_exp_avg, torch::Tensor state_exp_avg_sq, torch::Tensor state_slow_buf,
                             torch::Tensor x_expected,
                             RangerOptions option, RangerOptions default_option)
{

    x.mutable_grad() = grad;
    auto grp = torch::optim::OptimizerParamGroup({x});
    grp.set_options(std::make_unique<RangerOptions>(option));
    Ranger optimizer({grp}, default_option);

    // set state
    RangerParamState x_state;
    x_state.step(state_step);
    x_state.exp_avg(state_exp_avg);
    x_state.exp_avg_sq(state_exp_avg_sq);
    x_state.slow_buffer(state_slow_buf);
    optimizer.set_state(x, x_state);

    // run step
    optimizer.step();

    std::cout << "grad norm = " << torch::norm(grad) << std::endl;
    std::cout << "x norm = " << torch::norm(x) << std::endl;
    std::cout << "x_expected norm = " << torch::norm(x_expected) << std::endl;
    std::cout << "difference norm = " << torch::norm(x - x_expected) << std::endl;
}

void test_ranger_step()
{
    std::ifstream inf;
    inf.open("/home/mc/sidework/nnchess/tdlambda-nnue/tdlambda-py/build/grad_central_tests/step_test_1.txt");

    auto ranger_option = RangerOptions().betas({0.9, 0.999}).eps(1.0e-7);

    for (int sample_id = 0; sample_id < 3; sample_id++) {
        std::cout << "------------------------> Sample " << sample_id << std::endl;

        for (int group_id = 0; group_id < 8; group_id++)
        {
            std::cout << "========================> Group " << group_id << std::endl;

            // input
            torch::Tensor x = load_txt_tensor(inf);
            torch::Tensor x_grad = load_txt_tensor(inf);

            // state
            int state_step = load_txt_int(inf);
            torch::Tensor state_exp_avg = load_txt_tensor(inf);
            torch::Tensor state_exp_avg_sq = load_txt_tensor(inf);
            torch::Tensor state_slow_buf = load_txt_tensor(inf);

            // expected
            torch::Tensor x_expected = load_txt_tensor(inf);

            RangerOptions group_option(ranger_option);
            const double LR = 1e-3;
            if (group_id < 2) { // input
                group_option.lr(LR).gc_dim(0);
            } else if (group_id >= 6) { // output
                group_option.lr(LR / 10.0);
            } else {
                group_option.lr(LR);
            }

            test_step_single_param_(x, x_grad,
                                    state_step, state_exp_avg, state_exp_avg_sq, state_slow_buf,
                                    x_expected, group_option, ranger_option);
        }
    }

}