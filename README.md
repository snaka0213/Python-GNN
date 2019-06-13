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
  * `testReadout`:
    `readout` operates correctlly iff outputs in the last `Case 3: OK`.
  * `testFunctions`:
    outputs graphs of `loss` by gradient descent method.
  * `testSgd`:
    validates of the model learned via *Stochastic Gradient Descent*.
    * `train/` folder is assumed to be in the same directory.
    * Each train data is the following form:

      * `*_graph.txt`
      ```
      n # Number of nodes of graph
      a_11 a_12 ... a_1n # a_{ij} = 1 if there exists an edge between node i and j,
      a_21 a_22 ... a_2n # otherwise a_{ij} = 0.
      ...
      a_n1 a_n2 ... a_nn
      ```

      * `*_label.txt`
      ```
      <label of *_graph.txt> # 0 or 1
      ```

  * `testAdam`:
    validates the model which is learned via `Adam` (-> see **References** ).

## How it Works
  * `hyperParameters`:
    consists of all hyperParameters used in `functions` and `sgd`.
  * `myClass`:
    defines `Case` class and `Theta` class.
      * `Case`: used in `testReadout` and `testFunctions`,
      a pair of input and readout.
      * `Theta`: parameter set of classifier.
  * `readout`: defines `aggregate` and `readout`.
  * `functions`: defines `loss` function and `grad_loss`.
  * `sgd`: defines machine leaning by (momentum) *Stochastic Gradient Descent*.
  * `adam`: defines `Adam` (-> see **References** ).

## References
  * GNN: How Powerful are Graph Neural Networks?
  (https://arxiv.org/abs/1810.00826)
  * Adam: Adam: A Method for Stochastic Optimization
  (https://arxiv.org/abs/1412.6980)
