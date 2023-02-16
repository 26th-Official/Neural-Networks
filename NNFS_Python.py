# %%
import numpy as np
import nnfs
from nnfs.datasets import spiral_data,vertical_data

nnfs.init()

# %%
x,y = vertical_data(samples=100, classes=3)
x[:10]

# %%
class Dense_Layer():
    def __init__(self,n_input,n_neurons):
        self.weights = np.random.randn(n_input,n_neurons)
        self.biases = np.zeros((1,n_neurons))
        
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases
        
    def backward(self,dvalues):
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True)
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dinputs = np.dot(dvalues,self.weights.T)

# %%
class ReLU():
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
        
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# %%
class Softmax():
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        self.output = exp_values/np.sum(exp_values,axis=1,keepdims=True)
        
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - \
            np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
            single_dvalues)

# %%
class Categorical_Cross_Entropy():
    def forward(self,inputs,targets):
        clipped_input = np.clip(inputs,1e-7,(1-1e-7))
        if (len(targets.shape) == 1):
            confidence = np.array(inputs)[range(len(inputs)),targets]
        elif (len(targets.shape) == 2):
            confidence = np.sum(np.array(inputs)*targets)
        
        loss = -np.log(confidence)
        self.output = np.mean(loss)
        
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs/samples

# %%
def Accuracy_Calculate(inputs,targets):
    confidences = np.argmax(inputs,axis=1)
    if (len(targets.shape) == 2):
        targets = np.argmax(targets,axis=1)
        
    accuracy = np.mean(confidences==targets)
    return accuracy

# %%
dense1 = Dense_Layer(2,3)
dense2 = Dense_Layer(3,3)
activation1 = ReLU()
activation2 = Softmax()
loss = Categorical_Cross_Entropy()

dense1.forward(x)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])

loss.forward(activation2.output,y)
accuracy = Accuracy_Calculate(activation2.output,y)

print(loss.output)
print(accuracy)

# %%
dense1 = Dense_Layer(2,3)
dense2 = Dense_Layer(3,3)
activation1 = ReLU()
activation2 = Softmax()
loss = Categorical_Cross_Entropy()

# %%
best_loss = 999999
best_weight_1 = dense1.weights.copy()
best_weight_2 = dense2.weights.copy()
best_bias_1 = dense1.biases.copy()
best_bias_2 = dense2.biases.copy()

for i in range(20000):
    dense1.weights += 0.05 * np.random.randn(2,3)
    dense2.weights += 0.05 * np.random.randn(3,3)
    dense1.biases += 0.05 * np.random.randn(1,3)
    dense2.biases += 0.05 * np.random.randn(1,3)

    dense1.forward(x)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss.forward(activation2.output,y)
    accuracy = Accuracy_Calculate(activation2.output,y)
    
    if loss.output < best_loss:
        print(f"trial - {i}, Loss : {loss.output} , Accuracy : {accuracy}")
        best_loss = loss.output
        best_weight_1 = dense1.weights.copy()
        best_weight_2 = dense2.weights.copy()
        best_bias_1 = dense1.biases.copy()
        best_bias_2 = dense2.biases.copy()
        
    else:
        dense1.weights = best_weight_1.copy()
        dense2.weights = best_weight_2.copy()
        dense1.biases = best_bias_1.copy()
        dense2.biases = best_bias_2.copy()


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



