#pragma once
#ifndef RANGER_H
#define RANGER_H

#include <torch/torch.h>
#include <utility>
#include <vector>
#include <stdexcept>
#include <string>

struct TORCH_API RangerOptions : public torch::optim::OptimizerCloneableOptions<RangerOptions>
{
    RangerOptions(double lr = 1e-3);
    RangerOptions(const RangerOptions &other);
/*
        double lr = 1e-3,                                                                        // lr
        double alpha = 0.5, int k = 6, int N_sma_threshhold = 5,                                 // Ranger options
        double beta_1 = 0.95, double beta_2 = 0.999, double eps = 1e-5, double weight_decay = 0, // Adam options
        // Gradient centralization on or off, applied to conv layers only or conv + fc layers
        bool use_gc = true, bool gc_conv_only = false, bool gc_loc = true
 */
    // lr
    TORCH_ARG(double, lr) = 1e-3;
    // Ranger options
    TORCH_ARG(double, alpha) = 0.5;
    TORCH_ARG(int, k) = 6;
    TORCH_ARG(int, N_sma_threshhold) = 5;
    // Adam options
    typedef std::tuple<double, double> betas_t;
    TORCH_ARG(betas_t, betas) = std::make_tuple(0.95, 0.999);
    TORCH_ARG(double, eps) = 1e-5;
    TORCH_ARG(double, weight_decay) = 0;
    // Gradient centralization options
    TORCH_ARG(bool, use_gc) = true;
    TORCH_ARG(bool, gc_conv_only) = false;
    TORCH_ARG(bool, gc_loc) = true;
    TORCH_ARG(int, gc_dim) = 0;

public:
    void serialize(torch::serialize::InputArchive &archive) override;
    void serialize(torch::serialize::OutputArchive &archive) const override;
    TORCH_API friend bool operator==(
        const RangerOptions &lhs,
        const RangerOptions &rhs);
    ~RangerOptions() override = default;
    double get_lr() const override;
    void set_lr(const double lr) override;
};

struct TORCH_API RangerParamState
    : public torch::optim::OptimizerCloneableParamState<RangerParamState> {
  TORCH_ARG(int64_t, step) = 0;
  TORCH_ARG(torch::Tensor, exp_avg);
  TORCH_ARG(torch::Tensor, exp_avg_sq);
  TORCH_ARG(torch::Tensor, slow_buffer);

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const RangerParamState& lhs,
      const RangerParamState& rhs);
  ~RangerParamState() override = default;
};

struct RangerBufferStruct {
public:
    int step;
    double N_sma;
    double step_size;
};

class TORCH_API Ranger : public torch::optim::Optimizer
{
    RangerOptions hyperparams;
public:
    explicit Ranger(std::vector<torch::optim::OptimizerParamGroup> params_group,
                    RangerOptions defaults = {}) : Optimizer(std::move(params_group),
                                                             std::make_unique<RangerOptions>(defaults))
    {
        hyperparams = defaults;

        // parameter checks
        if (!(0.0 <= hyperparams.alpha() && hyperparams.alpha() <= 1.0)) {
            //raise ValueError(f'Invalid slow update rate: {alpha}')
            throw std::invalid_argument("Invalid slow update rate: " + std::to_string(hyperparams.alpha()));
        }
        if (!(hyperparams.k() >= 1)) {
            //raise ValueError(f'Invalid lookahead steps: {k}')
            throw std::invalid_argument("Invalid lookahead steps: " + std::to_string(hyperparams.k()));
        }
        if (!(hyperparams.lr() > 0)) {
            //raise ValueError(f'Invalid Learning Rate: {lr}')
            throw std::invalid_argument("Invalid Learning Rate: " + std::to_string(hyperparams.lr()));
        }
        if (!(hyperparams.eps() > 0)) {
            //raise ValueError(f'Invalid eps: {eps}')
            throw std::invalid_argument("Invalid eps: " + std::to_string(hyperparams.eps()));
        }

        // parameter comments:
        // beta1 (momentum) of .95 seems to work better than .90...
        // N_sma_threshold of 5 seems better in testing than 4.
        // In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        // prep defaults and init torch.optim base
        /*
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                        N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay,
                        gc_dim=None)
        super().__init__(params, defaults)
        */

        // adjustable threshold
        ////this->N_sma_threshhold = N_sma_threshhold;

        // look ahead params
        ////this->alpha = alpha;
        ////this->k = k;

        // radam buffer for state
        // this->radam_buffer = [[None, None, None] for ind in range(10)]

        // gc on or off
        ////this->gc_loc = gc_loc;
        ////this->use_gc = use_gc;
        ////this->gc_conv_only = gc_conv_only;

        // level of gradient centralization
        // this->gc_gradient_threshold = 3 if gc_conv_only else 1

        std::cout << "Ranger optimizer loaded. \nGradient Centralization usage = " << hyperparams.use_gc() << std::endl;
        if (hyperparams.use_gc() && hyperparams.gc_conv_only() == false) {
            std::cout << "GC applied to both conv and fc layers" << std::endl;
        } else if (hyperparams.use_gc() && hyperparams.gc_conv_only() == true) {
            std::cout << "GC applied to conv layers only" << std::endl;
        }
    }
    // explicit Ranger(std::vector<Tensor> params, RangerOptions defaults = {})
    //     : Ranger({torch::optim::OptimizerParamGroup(std::move(params))}, defaults) {}
    //  void __setstate__(self, state);
    torch::Tensor step(LossClosure closure = nullptr) override;
    void show_hypers();
};

#endif // #define RANGER_H

/*
                params, lr=1e-3,                       # lr
                 alpha=0.5, k=6, N_sma_threshhold=5,           # Ranger options
                 betas=(.95, 0.999), eps=1e-5, weight_decay=0,  # Adam options
                 # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                 use_gc=True, gc_conv_only=False, gc_loc=True) {
*/

void test_create_ranger_optimizer();
void test_centralized_gradient();
void test_compute_buffer();
void test_ranger_step();
