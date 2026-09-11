path = './network-1/data/mnist.npz'
import numpy as np

import matplotlib.pyplot as  plt
'''
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


'''
with np.load(path) as f:
   x_train, y_train = f['x_train'], f['y_train']
   x_test, y_test = f['x_test'], f['y_test']


#x_train, y_train = load_mnist('./network-1/data/fashion', kind='train')
#x_test, y_test = load_mnist('./network-1/data/fashion', kind='t10k')
#print(x_train[0].shape, 'hehehehe')

# Flatten the array
# makes 28x28 into 1x28*28 #firstlayerdone #typeshit #notevenclose #needtotransposeIthink
def flatenize(arr):
    return arr.flatten()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# need a 784x17 matrix with weights to multiply by the 1x784 first layer matrix to get a 1x17 matrix I hope
#each row belongs to a different second layer neuron typeshi


def activation(weights, bias, data):

 
    activation_vector = sigmoid(np.matmul(weights, data) + bias)

    

    return activation_vector

def cost(prediction_array, actual):

    comp_arr = np.zeros(10)
    comp_arr[actual] = 1

    sub_array = prediction_array - comp_arr
    cost = sub_array@sub_array
    return cost

def sigmoid_dir(x):
    dir = np.exp(-x) / (1 + np.exp(-x))**2
    
    return dir


def z_act(weights, bias, data):

    z_vector = np.matmul(weights, data) + bias
    return z_vector



def make_prediction(x, weights, biases):

    layer_1_act = flatenize(x)/255.0
    
    layer_1_weights = weights[0]
    layer_1_bias = biases[0]

    layer_2_weights = weights[1]
    layer_2_bias = biases[1]

    layer_3_weights = weights[2]
    layer_3_bias = biases[2]
    
    layer_2_act = activation(layer_1_weights,layer_1_bias,layer_1_act)
    layer_3_act = activation(layer_2_weights,layer_2_bias, layer_2_act)
    layer_4_act = activation(layer_3_weights,layer_3_bias, layer_3_act)

    z_act_2 = z_act(layer_1_weights,layer_1_bias,layer_1_act)
    z_act_3 = z_act(layer_2_weights,layer_2_bias, layer_2_act)
    z_act_4 = z_act(layer_3_weights,layer_3_bias, layer_3_act)

    activations = [x/255.0,layer_2_act,layer_3_act, layer_4_act]
    z_vals = [z_act_2,z_act_3,z_act_4]

    return [activations, z_vals]



def grad_fn(weights, activations, y, z_vals):
    comp_arr = np.zeros(10)
    comp_arr[y] = 1
    
    dc_da4 = 2*(activations[3]- comp_arr)
    da4_dz4 = sigmoid_dir(z_vals[2])
    dz4_dw3 = activations[2]

    dc_dz4 = dc_da4*da4_dz4
    dc_dw3 = np.outer(dc_dz4,dz4_dw3)
#all up to this point it w3 dirivitive

    dc_dz4 = dc_da4*da4_dz4
    dc_da3 = np.matmul(weights[2].T, dc_dz4 )
    da3_dz3 = sigmoid_dir(z_vals[1])
    dc_dz3 = dc_da3*da3_dz3
    dz3_dw2 = activations[1]
    dc_dw2 = np.outer(dc_dz3,dz3_dw2)
    #now onto layer 1 weights
    dc_da2 = np.matmul(weights[1].T,dc_dz3)
    da2_dz2 = sigmoid_dir(z_vals[0])
    dc_dz2 = dc_da2*da2_dz2
    dz2_dw1 = activations[0]
    dc_dw1 = np.outer(dc_dz2,dz2_dw1)
    #now onto biases

    dc_db4 = dc_dz4
    dc_db3 = dc_dz3
    dc_db2 = dc_dz2
    
    return [dc_dw1, dc_dw2, dc_dw3, dc_db2, dc_db3, dc_db4]




def batchItUp(x_cut, y_cut, weights, bias):

    combined_grad = [np.zeros((17,784)), np.zeros((17,17)), np.zeros((10,17)), np.zeros((17,)), np.zeros((17,)), np.zeros((10,))]
    
    batch_size = len(x_cut)
    for i in range(batch_size):

        predic = make_prediction(x_cut[i], weights, bias)
        activations = predic[0]
        z_vals = predic[1]
        grad = grad_fn(weights, activations, y_cut[i], z_vals)
        
        for e in range(len(combined_grad)):
            combined_grad[e] = combined_grad[e]+grad[e]
        
    final_grad = [None] * 6
    for l in range(len(combined_grad)):

        final_grad[l] = combined_grad[l]/batch_size
    
    return final_grad



def train(x, y, starting_vals, epochs, bach_size, step_size):

    if len(starting_vals)<10:

        starting_vals = randomise()

    current_vals = starting_vals
    index = 0
    total_baches = len(x)//bach_size

    for p in range(epochs):
        for i in range(total_baches):
            
            weights = current_vals[0]
            bias = current_vals[1]
            index = i*bach_size
            to = index + bach_size
            x_cut = x[index:to]
            y_cut = y[index:to]
            
            avg_grad = batchItUp(x_cut, y_cut, weights, bias )

            for e in range(len(current_vals)):
                for poop in range(len(current_vals[e])):
                    
                    current_vals[e][poop] -= (avg_grad[poop+3*e]*step_size)

        pred_check = make_prediction(x[0], current_vals[0], current_vals[1])
        print(cost(pred_check[0][3], y[0]), 'epoch num:', p)
        perm = np.random.permutation(len(x_train))
        x = x[perm]
        y = y[perm]
    return current_vals


def randomise():

    layer_1_weights = np.random.uniform(-0.1, 0.1, size=(17,784))
    layer_1_bias = np.zeros((17))

    layer_2_weights = np.random.uniform(-0.1, 0.1, size=(17,17))
    layer_2_bias = np.zeros((17))

    layer_3_weights = np.random.uniform(-0.1, 0.1, size=(10,17))
    layer_3_bias = np.zeros((10))

    ret_vals = [[layer_1_weights, layer_2_weights, layer_3_weights], [layer_1_bias, layer_2_bias, layer_3_bias]]
    
    return(ret_vals)

def save_train(trained_model, as_what):

    np.savez(
        as_what,
        w3 = trained_model[0][0],
        w2 = trained_model[0][1],
        w1 = trained_model[0][2],
        b3 = trained_model[1][0],
        b2 = trained_model[1][1],
        b1 = trained_model[1][2]
    )
    print('model saved')


'''

starting_vals = []
epochs = 57
bach_size = 17
step_size = 0.1

trained_model = train(x_train, y_train, starting_vals, epochs, bach_size, step_size)
print(trained_model)
save_train(trained_model)


'''

def load_data(which_one):
    data = np.load(which_one)

    weights = [
        data["w3"],
        data["w2"],
        data["w1"]
    ]
    biases = [
        data["b3"],
        data["b2"],
        data["b1"]
    ]
    return [weights, biases]
'''
model = load_data()
weights = model[0]
biases = model[1]
print(y_test)

correct = 0
for i in range(len(x_test)):
    layer_4_act = make_prediction(x_test[i], weights, biases)[0][3]
    prediction = np.argmax(layer_4_act)
    print(y_test[i], prediction)
    if prediction == y_test[i]:
        correct+=1
    else:
        print('blalba')
    
        
accuracy = correct/len(x_test)
print(accuracy, correct)

test_i = 1
layer_4_act = make_prediction(x_test[test_i], weights, biases)[0][3]
prediction = np.argmax(layer_4_act)




fruits = [0,1,2,3,4,5,6,7,8,9]
sales = layer_4_act
print(cost(layer_4_act, y_test[1]))
print(y_test[test_i])
# Create bar graph
plt.bar(fruits, sales)

# Add labels and title
plt.xlabel('digit')
plt.ylabel('prediction')
plt.title('Fruit Sales')

# Display the plot
plt.show()

plt.figure(figsize=(10, 3))

plt.subplot(1, 4, 1)
#shaped = np.reshape(x_test[test_i], (28,28))
plt.imshow(x_test[test_i], cmap="gray")
plt.title(f"Label: {y_test[test_i]}")
plt.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 3))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(x_test[i], cmap="gray")
    plt.title(f"Label: {y_test[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

'''

