# banknote_bnn.py
# Banknote classification
# Anaconda3 5.2.0 (Python 3.6.5), PyTorch 1.0.0
#
# raw data looks like:
#  4.5459, 8.1674, -2.4586, -1.4621, 0
# -1.3971, 3.3191, -1.3927, -1.9948, 1
#  0 = authentic, 1 = fake

import numpy as np
import torch as T

# ------------------------------------------------------------

class Batcher:
  def __init__(self, num_items, batch_size, seed=0):
    self.indices = np.arange(num_items)
    self.num_items = num_items
    self.batch_size = batch_size
    self.rnd = np.random.RandomState(seed)
    self.rnd.shuffle(self.indices)
    self.ptr = 0

  def __iter__(self):
    return self

  def __next__(self):
    if self.ptr + self.batch_size > self.num_items:
      self.rnd.shuffle(self.indices)
      self.ptr = 0
      raise StopIteration  # exit calling for-loop
    else:
      result = self.indices[self.ptr:self.ptr+self.batch_size]
      self.ptr += self.batch_size
      return result

# ------------------------------------------------------------

def akkuracy(model, data_x, data_y):
  # data_x and data_y are numpy array-of-arrays matrices
  X = T.Tensor(data_x)
  Y = T.ByteTensor(data_y)   # a Tensor of 0s and 1s
  oupt = model(X)            # a Tensor of floats
  pred_y = oupt >= 0.5       # a Tensor of 0s and 1s
  num_correct = T.sum(Y==pred_y)  # a Tensor
  acc = (num_correct.item() * 100.0 / len(data_y))  # scalar
  return acc

# ------------------------------------------------------------

class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hid1 = T.nn.Linear(4, 8)  # 4-(8-8)-1
    self.hid2 = T.nn.Linear(8, 8)
    self.oupt = T.nn.Linear(8, 1)

    T.nn.init.xavier_uniform_(self.hid1.weight)  # glorot
    T.nn.init.zeros_(self.hid1.bias)
    T.nn.init.xavier_uniform_(self.hid2.weight)  # glorot
    T.nn.init.zeros_(self.hid2.bias)
    T.nn.init.xavier_uniform_(self.oupt.weight)  # glorot
    T.nn.init.zeros_(self.oupt.bias)

  def forward(self, x):
    z = T.tanh(self.hid1(x))  # or T.nn.Tanh() !!
    z = T.tanh(self.hid2(z))
    z = T.sigmoid(self.oupt(z))  # BCELoss() doesn't apply
    return z

# ------------------------------------------------------------

def main():
  # 0. get started
  print("\nBanknote authentication using PyTorch \n")
  T.manual_seed(1)
  np.random.seed(1)

  # 1. load data
  print("Loading Banknote data into memory \n")
  train_file = "./Data/banknote_norm_train.txt"
  test_file = "./Data/banknote_norm_test.txt"

  train_x = np.loadtxt(train_file, delimiter='\t',
    usecols=[0,1,2,3], dtype=np.float32)
  train_y = np.loadtxt(train_file, delimiter='\t',
    usecols=[4], dtype=np.float32, ndmin=2)
  test_x = np.loadtxt(test_file, delimiter='\t', 
    usecols=[0,1,2,3], dtype=np.float32)
  test_y =np.loadtxt(test_file, delimiter='\t',
    usecols=[4], dtype=np.float32, ndmin=2)

  # 2. define model
  print("Creating 4-(8-8)-1 binary NN classifier \n")
  net = Net()

  # 3. train model
  net = net.train()  # set training mode
  lrn_rate = 0.01
  bat_size = 16
  loss_func = T.nn.BCELoss() 
  optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)

  max_epochs = 100
  n_items = len(train_x)
  batcher = Batcher(n_items, bat_size)

# ------------------------------------------------------------

  print("Starting training")
  for epoch in range(0, max_epochs):
    if epoch > 0 and epoch % (max_epochs/10) == 0:
      print("epoch = %6d" % epoch, end="")
      print("  batch loss = %7.4f" % loss_obj.item(), end="")
      acc = akkuracy(net, train_x, train_y)
      print("  accuracy = %0.2f%%" % acc) 

    for curr_bat in batcher:
      X = T.Tensor(train_x[curr_bat])
      Y = T.Tensor(train_y[curr_bat])
      optimizer.zero_grad()
      oupt = net(X)
      loss_obj = loss_func(oupt, Y)
      loss_obj.backward()
      optimizer.step()
  print("Training complete \n")

  # 4. evaluate model
  net = net.eval()  # set eval mode
  acc = akkuracy(net, test_x, test_y)
  print("Accuracy on test data = %0.2f%%" % acc)

  # 5. save model
  print("Saving trained model \n")
  path = "./Models/banknote_model.pth"
  T.save(net.state_dict(), path)

  # model = Net()
  # model.load_state_dict(T.load(path))

# ------------------------------------------------------------

  # 6. make a prediction 
  train_min_max = np.array([
    [-7.0421, 6.8248],
    [-13.7731, 12.9516],
    [-5.2861, 17.9274],
    [-7.8719, 2.1625]], dtype=np.float32)

  unknown_raw = np.array([[1.2345, 2.3456, 3.4567, 4.5678]],
    dtype=np.float32)
  unknown_norm = np.zeros(shape=(1,4), dtype=np.float32)
  for i in range(4):
    x = unknown_raw[0][i]
    mn = train_min_max[i][0]  # min
    mx = train_min_max[i][1]  # max
    unknown_norm[0][i] = (x - mn) / (mx - mn)

  np.set_printoptions(precision=4)
  print("Making prediction for banknote: ")
  print(unknown_raw)
  print("Normalized to:")
  print(unknown_norm)

  unknown = T.Tensor(unknown_norm)  # to Tensor
  raw_out = net(unknown)       # a Tensor
  pred_prob = raw_out.item()   # scalar, [0.0, 1.0]

  print("\nPrediction prob = %0.4f " % pred_prob)
  if pred_prob < 0.5:
    print("Prediction = authentic")
  else:
    print("Prediction = forgery")

if __name__=="__main__":
  main()
