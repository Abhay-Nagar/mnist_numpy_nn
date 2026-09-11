from main import train, save_train, load_data, make_prediction
import numpy as np

new_model_name = "insert_name_here.npz"

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

x_train, y_train = load_mnist('./network-1/data/fashion', kind='train')
x_test, y_test = load_mnist('./network-1/data/fashion', kind='t10k')
print(x_train.shape, 'hehehehe')

starting_vals = []
epochs = 5
bach_size = 17
step_size = 0.1

trained_model = train(x_train, y_train, starting_vals, epochs, bach_size, step_size)
print(trained_model)
save_train(trained_model,new_model_name)

model = load_data(new_model_name)
weights = model[0]
biases = model[1]


correct = 0
for i in range(len(x_test)):
    layer_4_act = make_prediction(x_test[i], weights, biases)[0][3]
    prediction = np.argmax(layer_4_act)
    print(y_test[i], prediction)
    if prediction == y_test[i]:
        correct+=1
    
    
        
accuracy = correct/len(x_test)
print(accuracy, correct)