
from main import load_data, make_prediction
import numpy as np
import numpy as np





path = './network-1/data/mnist.npz'
with np.load(path) as f:
  x_train, y_train = f['x_train'], f['y_train']
  x_test, y_test = f['x_test'], f['y_test']

model = load_data("model.npz")
weights = model[0]
biases = model[1]


correct = 0
for i in range(len(x_test)):
  layer_4_act = make_prediction(x_test[i], weights, biases)[0][3]
  prediction = np.argmax(layer_4_act)
  
  if prediction == y_test[i]:
      correct+=1
  
  
      
accuracy = correct/len(x_test)
print("mnist accuracy is ", accuracy)


#test fashion model
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


x_test, y_test = load_mnist('./network-1/data/fashion', kind='t10k')

model = load_data("model_fashion.npz")
weights = model[0]
biases = model[1]


correct = 0
for i in range(len(x_test)):
    layer_4_act = make_prediction(x_test[i], weights, biases)[0][3]
    prediction = np.argmax(layer_4_act)
    
    if prediction == y_test[i]:
        correct+=1
    
    
        
accuracy = correct/len(x_test)
print("mnist fashion accuracy is ", accuracy)
