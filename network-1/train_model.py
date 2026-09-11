path = './network-1/data/mnist.npz'
import numpy as np

from main import train, save_train, make_prediction, load_data

new_model_name = "insert_name_here.npz"

with np.load(path) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

starting_vals = []
epochs = 1
bach_size = 17
step_size = 0.1

trained_model = train(x_train, y_train, starting_vals, epochs, bach_size, step_size)
save_train(trained_model, new_model_name)

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