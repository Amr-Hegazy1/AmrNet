# AmrNet: Deep Learning Library

AmrNet is a lightweight deep learning library designed for simplicity and ease of use. It provides a set of basic tools to help you quickly build and train neural networks.

<!-- ## Features

- **Simple API:** Easily define, train, and evaluate neural networks with just a few lines of code.
- **Modularity:** Construct networks using pre-built layers or create custom layers for more flexibility.
- **Optimizers:** Choose from a variety of gradient descent optimizers to train your models.
- **Example Usage:** Check out the `examples` directory for code snippets and sample projects. -->

## Installation

```bash
pip install amrnet==0.1.0
```

## Implemented Features

- [Tensors](./amrnet/tensor.py)
- [Layers](./amrnet/layers.py) :

    1. Linear
    2. ReLU
    3. Tanh
    4. LeakyReLU
    5. Sigmoid
    6. Softmax
    7. Dropout
    <br></br>
- [Neural Networks](./amrnet/nn.py)
- [Loss Functions](./amrnet/loss.py) :

    1. MSE
    2. TSE
    3. MAE
    4. LogCosh
    5. Huber
    <br></br>
- [Optimizers](./amrnet/optim.py) :

    1. SGD
    <br></br>
- [Data Utilities](./amrnet/data.py)
- [Training Utilities](./amrnet/train.py)



## Usage

### Creating a Model

```python
from amrnet.nn import NeuralNet
from amrnet.layers import Linear, Tanh,  ReLU

net = NeuralNet([
    Linear(input_size, hidden_size),
    Tanh(),
    Linear(hidden_size, output_size)
])
```

### Training the Model

```python
from amrnet.train import train

train(net, inputs, targets, num_epochs, data_iterator, loss, optimizer)
```

### Predicting

```python
predicted = net.forward(x)
```

## Examples

Check out the [examples](./examples/) directory for a variety of different projects using AmrNet.

## License

[MIT](LICENSE)


## TODO
- [ ] Add more layers
- [ ] Add more optimizers
- [ ] Add more loss functions
- [ ] Add more data utilities
- [ ] Add more training utilities
- [ ] Add more examples
- [ ] Add more tests
- [ ] Add more documentation


