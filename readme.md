# NNUE Cpp Trainer

**NNUE Cpp Trainer** is an NNUE network trainer implemented with PyTorch C++ frontend (LibTorch). Similar to the [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) in which a Stockfish NNUE trainer is implemented in Python with [Pytorch]() library, this project is a C++ derivation of nnue-pytorch, aiming to provide a more C++ friendly implementation of trainer for any future research that is related to NNUE technique. The key difference of  this implementation from the original trainer by Nodchip is that, this project is still heavily relying on Pytorch (LibTorch).

Currently the project only supports the original network architecture (the 2020 version), which means that the trained network can only be used on Stockfish 13 or earlier versions of Stockfish engines.


## Compiling from the Source

To compile the source code, you need to have the following tools and libraries installed:
* CMake
* LibTorch

Under the root directory of project, run `cd src`, modify the run_cmake.sh with your LibTorch installed path:
```
cmake -DCMAKE_PREFIX_PATH=<YOUR-LIBTORCH-PATH> ..
```
Then run
```
mkdir build
cd build
../run_cmake.sh
```
This will trigger cmake running. Once cmake is done, just run `make` to start the actual compiling.

## Run

A 600-epoch training procedure will start after running the following command:
```
./cpp-trainer
```


## Sources of Training Data

I am using the training data provided by the official nnue-pytorch project. See [this link](https://github.com/official-stockfish/nnue-pytorch/wiki/Training-datasets) for more details about the available training data. The following items are just copied from the linked page:  

> * **large_gensfen_multipvdiff_100_d9.binpack** - the original depth 9 dataset generated with Stockfish. It's still ok, but superseded by the more recent one listed below.
> * **data_d9_2021_09_02.binpack** - the most recent dataset generated with Stockfish; depth 9 still seems to be the sweet spot.
> * **nodes5000pv2_UHO.binpack** - a recent attempt to use fixed nodes instead of fixed depth for data generation using Stockfish. It produces the best playing nets for the UHO books. and at least on par with the depth 9 dataset for others.
> * **training_data.binpack** - the first good dataset derived from Lc0. It was used to beat the then master Stockfish net.
> * **T60T70wIsRightFarseer.binpack** - a mix of various datasets, including Lc0 T60, T70 data, Stockfish self-play data from openings it usually gets wrong, and some more converted Lc0 data from Farseer. It is currently one of the best datasets available.
> * **dfrc_n5000.binpack** - new data generated with stockfish at 5000 nodes per move, from DFRC opening book. Not used in isolation.
> * **Leela-dfrc_n5000.binpack** - made by running `python3 interleave_binpacks.py T60T70wIsRightFarseer.binpack dfrc_n5000.binpack Leela-dfrc_n5000.binpack`. Used for retraining the network.

However, you can always generate the data file by yourself.

So far, the data file used to archive the best network in this project (NOT the official network) is `large_gensfen_multipvdiff_100_d9.binpack`.

## Play Strength

With the "40 moves/5 minutes" time control, the trained network is only 20 ELO weaker than the standard network released in Stockfish 13. Here is the output given by BayesELO:

| Rank Name | Elo | + | - | games | score | oppo. | draws |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  1 stockfish13 | 	9 |	5 |	5 | 3200 |  54%	| -9 |  87% |
|  2 stockfish13-cpp-trainer |	-9	| 5	| 5  | 3200 |  46% | 9 |  87% |


## Dependencies

The source files under `src/lib` folder and `training_data_loader.h` are originally from nnue-pytorch project.

## TODO

The project is still in progress. More specifically, the following features has not been implemented yet:

* Taking parameters from main arguments;
* Add support to the tensorboard (for training progress monitoring);
* Add complete support to the FeatureSet (only the most basic feature set is supported so far, and many parameters are hard-coded);
* Complete the features in the serializer;

And most importantly, try to support the more recent network architectures applied in Stockfish after 13.
