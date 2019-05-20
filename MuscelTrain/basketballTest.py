import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

'''
statement：training(25047):  LeftHand----LeftPre----LeftArm----
                        RightHand----RightPre----RightArm----
                        label(0: non-basketball; 1: basketball)
           test(7000+)
'''

np.random.seed(1)

# read the data from file
def loadData(file_path, k = 18):
  with open(file_path) as file_object:
    lines = file_object.readlines()
    dataset =[]
    train = np.array([])
    label = np.array([])
    for line in lines:
      temp1 = line.strip('\n')
      temp2 = temp1.split(' ')
      dataset.append(temp2)
    for i in range(0,len(dataset)):
      for j in range(0, 19):
        dataset[i].append((float(dataset[i][j])))
      del(dataset[i][0:19])

    train=np.array(dataset)[:,:18]
    label=np.array(dataset)[:,18:]

    for i in range(0,label.shape[0]):
        if label[i][0] == 1:
            label[i][0] = 0
        else:
            label[i][0] = 1

    return train,label

def layer_sizes(train_data_shape, label_data_shape):
    n_x = train_data.shape[1]
    n_h = 36
    n_y = label_data.shape[1]
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    W1 = np.random.randn(n_x, n_h) * 0.01
    b1 = np.zeros((1,n_h))
    W2 = np.random.randn(n_h, n_y) * 0.01
    b2 = np.zeros((1,n_y))

    assert (W1.shape == (n_x, n_h))
    assert (b1.shape == (1, n_h))
    assert (W2.shape == (n_h, n_y))
    assert (b2.shape == (1, n_y))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(X, W1) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (X.shape[0], 1))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

def compute_cost(A2, Y, parameters):
    m = Y.shape[0]  # number of example

    logprobs = Y * np.log(A2.T) + (1 - Y) * np.log(1 - A2.T)
    cost = -1 / m * np.sum(logprobs)

    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
                             # E.g., turns [[17]] into 17
    assert (isinstance(cost, float))

    return cost


def backward_propagation(parameters, cache, X, Y):
    m = X.shape[0]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(A1.T, dZ2)
    db2 = 1 / m * np.sum(dZ2.T, axis=1, keepdims=True)
    db2 = db2.T
    dZ1 = np.dot(dZ2, W2.T) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(X.T, dZ1)
    db1 = 1 / m * np.sum(dZ1.T, axis=1, keepdims=True)
    db1 = db1.T

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

def update_parameters(parameters, grads, learning_rate=0.1):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def nn_model(X, Y, n_h, num_iterations=5, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)

        # Print the cost every 1000 iterations
        # if print_cost and i % 1000 == 0:
        print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)

    return predictions

if __name__ == '__main__':
    file_path = '/home/wby/tensorflow demo/datas/Basketball/train.txt'
    train_data, label_data = loadData(file_path)
    shape_train = train_data.shape
    shape_label = label_data.shape

    m = shape_train[0]   # training set size

    print_cost = False

    n_x, n_h, n_y = layer_sizes(train_data,label_data)
    parameters = initialize_parameters(n_x, n_h, n_y)

    '''for i in range(0, 5):
        A2, cache = forward_propagation(train_data, parameters)
        # print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))

        cost = compute_cost(A2, label_data, parameters)

        grads = backward_propagation(parameters, cache, train_data, label_data)

        parameters = update_parameters(parameters,grads)

        # if print_cost and i % 1000 == 0:
        print("Cost after iteration %i: %f" % (i, cost))
    '''
    '''print("dW1 = " + str(grads["dW1"]))
    print("db1 = " + str(grads["db1"]))
    print("dW2 = " + str(grads["dW2"]))
    print("db2 = " + str(grads["db2"]))
    '''
    parameters = nn_model(train_data,label_data,4,num_iterations=5,print_cost=False)

    predictions = predict(parameters,train_data)
    print('Accuracy: %d' % float(
        (np.dot(predictions.T, label_data) + np.dot(1 - predictions.T, 1 - label_data)) / float(label_data.size) * 100) + '%')

    # print("predictions mean = " + str(np.mean(predictions)))

    '''
    hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20]
    for i, n_h in enumerate(hidden_layer_sizes):
        parameters = nn_model(train_data,label_data,n_h,num_iterations=5,print_cost=False)
        predictions = predict(parameters, train_data)
        accuracy = float(np.dot(predictions.T, label_data) + np.dot(1 - predictions.T, 1 - label_data)) / float(label_data.size) * 100
        print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
    '''

    # test
    file_path = '/home/wby/tensorflow demo/datas/Basketball/testChange.txt'
    test_data, labeltest_data = loadData(file_path)
    my_predictions = predict(parameters, test_data)
    print(my_predictions)
    np.savetxt("/home/wby/tensorflow demo/datas/Basketball/result.txt", my_predictions, fmt='%s')