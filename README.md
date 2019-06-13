# Graph Neural Network

## Contents
```
├── LICENSE
├── README.md
└── src
    ├── adam.py
    ├── functions.py
    ├── hyperParameters.py
    ├── myClass.py
    ├── readout.py
    ├── sgd.py
    ├── testAdam.py
    ├── testFunctions.py
    ├── testReadout.py
    └── testSgd.py
```

## How to Use

## How it Works
  * `hyperParameters`:
    consists of all hyperParameters used in `functions` and `sgd`.
  * `myClass`:
    defines `Case` and `Theta`.
      * `Case` instances are used in `testReadout` and `testFunctions`, consists of a pair of input and output of readout.
      * `Theta` instance is the parameter set of learning model.
  * `readout`: defines `aggregate` and `readout`.
  * `functions`: defines `loss` function and `grad_loss`.
  * `sgd`: consists of Stochastic Gradient Descent and its momentum version.
  * `adam`: defines `Adam` (-> see **References** ).

## References
  * GNN:
  * Adam:
