import numpy as np

size_InpLayer = 900       # amount of inputs
size_HidLayer = 600       # amount of hidden neurons
size_OutLayer = 10       # amount of output neurons
lr = 0.05               # learning rate


weightLayer1 = np.load("weight_layer1.npy")
weightLayer2 = np.load("weight_layer2.npy")


def processing(layerIn, answer=None):
    print("input=", layerIn, answer)
    global size_InpLayer, size_HidLayer, size_OutLayer, weightLayer2, weightLayer1
    layerHid = {"in": [0 for _ in range(size_HidLayer)],  # one list for sum, one for output
                "out": [0 for _ in range(size_HidLayer)]}
    layerOut = {"in": [0 for _ in range(size_OutLayer)],  # one list for sum, one for output
                "out": [0 for _ in range(size_OutLayer)]}

    # calculate hidden layer
    for ind_HN in range(size_HidLayer):  # index of Hidden Neuron
        for ind_IN in range(size_InpLayer):  # index of Input Neuron
            # sum input*weight for every hidden neuron
            layerHid["in"][ind_HN] += layerIn[ind_IN] * weightLayer1[ind_HN][ind_IN]
        layerHid["in"][ind_HN] += weightLayer1[ind_HN][size_InpLayer]  # add bias
        layerHid["out"][ind_HN] = relu(layerHid["in"][ind_HN])  # use relu activation function

    # calculate output layer
    for ind_ON in range(size_OutLayer):
        for ind_HN in range(size_HidLayer):
            layerOut["in"][ind_ON] += layerHid["out"][ind_HN] * weightLayer2[ind_ON][ind_HN]
        layerOut["in"][ind_ON] += weightLayer2[ind_ON][size_HidLayer]
    layerOut["out"] = softmax(layerOut["in"])  # use softmax activation function

    if answer is not None:
        backprop(layerIn, layerOut, layerHid, answer)
    return layerOut["out"]


# calculate the softmax of a vector
def softmax(vector):
    e = np.exp(vector)
    return e / sum(e)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return max(0.001, x)


def backprop(layerIn, layerOut, layerHid, answer):
    global weightLayer2, weightLayer1, size_InpLayer, size_HidLayer, size_OutLayer, lr
    dE_dOin = [0 for _ in range(size_OutLayer)]

    # calculate new weight between hidden layer and output layer
    for o in range(size_OutLayer):
        dE_dOout = d_crossentropy(answer[0], layerOut["out"][o])
        dOout_dOin = d_softmax(layerOut["in"], o)
        dE_dOin[o] = dE_dOout * dOout_dOin
        print(layerOut["in"])
        print("ddddddd", dE_dOout, dOout_dOin)
        for k in range(size_HidLayer):
            dOin_dWko = layerHid["in"][k]
            weightLayer2[o][k] = weightLayer2[o][k] - lr*dE_dOin[o]*dOin_dWko
        weightLayer2[o][size_HidLayer] = weightLayer2[o][size_HidLayer] - lr*dE_dOin[o]     # change bias weight

    for k in range(size_HidLayer):
        dEtotal_dKout = sum([dE_dOin[i] * weightLayer2[i][k] for i in range(size_OutLayer)])   # dE_dOin * dOin_dKout
        dKout_dKin = d_sigmoid(layerHid["in"][k])
        for j in range(size_InpLayer):
            dKin_dWjk = layerIn[j]
            weightLayer1[k][j] = weightLayer1[k][j] - lr * dEtotal_dKout * dKout_dKin * dKin_dWjk
        # change bias weight
        weightLayer1[k][size_InpLayer] = weightLayer1[k][size_InpLayer] - lr * dEtotal_dKout * dKout_dKin

    np.save("weight_layer1.npy", weightLayer1)
    np.save("weight_layer2.npy", weightLayer2)

    with open('Errors.txt', 'a') as f:
        f.write(str(answer) + '\n')
        f.write(str(layerOut["out"]))
    print(weightLayer1)
    np.savetxt("weight_layer1.txt", weightLayer1)
    np.savetxt("weight_layer2.txt", weightLayer2)
 

def d_relu(x):
    return 1 if x > 0 else 0.001


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def d_softmax(vector, ind):
    proc_vec = vector.copy()                    # make copy of vector for not to change orig vector
    exp_sum = sum(np.exp(proc_vec))            # firstly count the sum of exp of all elements
    x = proc_vec.pop(ind)                       # get x[i] and delete it from vector
    exp_leftover_sum = sum(np.exp(proc_vec))   # count sum of other elements
    return np.exp(x)*exp_leftover_sum/(exp_sum*exp_sum)


def d_crossentropy(y, o):
    return o - y
