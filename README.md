### Experiments

#### Install the experiment requirements 
`pip install -r requirements.txt`


### Usage
Use linesearch in your code by adding the following script.

```python
import optimizers.sls
opt = sls.SGD(model.parameters())
for epoch in range(100):
      # create loss closure
      closure = lambda : torch.nn.CrossEntropyLoss()(model(X), y)

      # update parameters
      opt.zero_grad()
      loss = opt.step(closure=closure)
```

#### select algorithm
`python train.py --cuda 0 --batch 128 --dataset CIFAR100 --algorithm SGD+Armijo --dir 'your_directly'`

#### plot accuracy
`python plot_list/plot_accuracy --dir 'your_directly'`
