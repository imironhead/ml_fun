# Train a CNN to Predict Next Step of Conway's Game of Life

[https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life](Wiki: Conway's Game of Life)

![](/assets/life_copperhead.gif)

This is a definite **useless** predictor. But with literally infinite training data, I can make experiments on this.

## Experimented Model

### Smallest One:

```
input shape: 32x32x1
mini batch size: 128
learning rate: 0.00001
training steps >= 555,000
accuracy ~ 100%

conv_valid_k3f4 + relu
conv_same_k3f3 + relu
conv_same_k3f1 + sigmoid
```

### Fast Converge

```
input shape: 32x32x1
mini batch size: 128
learning rate: 0.001
training steps: 20,000
accuracy ~ 100%

conv_valid_k3f4 + relu
conv_same_k3f4 + relu
conv_same_k3f4 + relu
conv_same_k3f1 + sigmoid
```


### Others

```
input shape: 32x32x1
mini batch size: 128
learning rate: 0.0001
training steps: 40,000
accuracy ~ 100%

conv_valid_k3f4 + relu
conv_same_k3f4 + relu
conv_same_k3f1 + sigmoid
```

```
input shape: 32x32x1
mini batch size: 128
learning rate: 0.00001
training steps: 2,000,000
accuracy ~ 98%

conv_valid_k3f8 + relu
conv_same_k3f1 + sigmoid
```
