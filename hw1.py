import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d

'''
This code is to solve the optimal investment strategy problem by using the deep neural network (DNN) model to approximate the Bellman equation.
The optimal investment strategy is to maximize the expected utility of the terminal wealth.
The utility function is the exponential utility function, which is U(W_T) = -exp(-a*W_T)/a, where W_T is the terminal wealth and a is the risk aversion coefficient.
The investment strategy is to invest in two assets: the risk-free asset and the risky asset.
The risky asset has two possible return rates: A and B, with probabilities p and 1-p, respectively.
The risk-free asset has a constant return rate r.
The goal is to find the optimal investment strategy to maximize the expected utility of the terminal wealth.

The code consists of the following parts:
1. Define the DNN model to approximate the Bellman equation.
2. Define the utility function.
3. Train the DNN model to get the optimal value function.
4. Get the optimal path of investment strategy and wealth by using the trained DNN model.

Our homework is completed by CHEN ZILING and CHEN YUWEI. We work together on the main code, which means the file hw1.py. While CHEN YUWEI contributes to the unit test and integration test part, and CHEN ZILING contributes to the code comments.
'''
# set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# define the DNN model to approximate the bellman equation
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) # input layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # hidden layer
        self.fc3 = nn.Linear(hidden_dim, 1) # output layer
        self.relu = nn.ReLU() # activation function
    
    def forward(self, x):
        x = self.relu(self.fc1(x)) 
        x = self.relu(self.fc2(x)) 
        return self.fc3(x)
    
# define the utility function
def Utility_function(W_T, a):
    return -np.exp(-a * W_T) / a # exponential utility function

# train the DNN model to get the optimal value function
def train_DNN(T, a, p, A, B, r, W0, num_epochs=1000, lr=0.0005):
    # Calculate the possible maximum and minimum wealth range
    max_single_return = max(A, B, r)
    min_single_return = min(A, B, r)
    max_possible_W = W0 * (1 + max_single_return) ** T
    min_possible_W = W0 * (1 + min_single_return) ** T
    
    # generate a wealth sequence to cover all the possible circumstance of wealth
    W_values = np.linspace(min_possible_W, max_possible_W, num=100)
    max_W = max_possible_W  # set the maximum wealth
    min_W = min_possible_W # set the minimum wealth
    # normalize the t and wealth into [0,1] range to avoid the gradient explosion
    normalize = lambda t, W: np.array([t / T, (W - min_W) / (max_W - min_W)])

    # Use the DNN model to solve the optimal question
    model = ValueNetwork(input_dim=2, hidden_dim=16) # initialize the DNN model
    optimizer = optim.Adam(model.parameters(), lr=lr) # set the optimizer
    criterion = nn.MSELoss() # set the loss function

    V_optimal = {} # store the optimal value function

    # get through the wealth sequence and time horizen to get optimal value and corresponding action policy
    for t in reversed(range(T)):
        V_optimal[t] = {} # store the optimal value at time t
        for W_t in W_values: 
            best_value = -np.inf 
            for x_t in np.linspace(0, 1, 11): 
                W_next_a = x_t * W_t*(1+A) + (1-x_t)*W_t*(1+r) # get the wealth value at next time step with action a
                W_next_b = x_t * W_t*(1+B) + (1-x_t)*W_t*(1+r) # get the wealth value at next time step with action b
                
                # if it is the terminal time step
                if t == T-1: 
                    expected = p*Utility_function(W_next_a, a) + (1-p)*Utility_function(W_next_b, a) # get the expected return
                else:
                    # Using dynamic interpolation methods to get wealth value at next time step to avoid data overflow
                    if W_next_a > max_W: W_next_a = max_W  
                    if W_next_b > max_W: W_next_b = max_W
                    if W_next_a < min_W: W_next_a = min_W
                    if W_next_b < min_W: W_next_b = min_W
                    # interpolate the value function at next time step. Beacause the value function is not continuous, we use linear interpolation to get the value.
                    interp_V = interp1d(W_values, list(V_optimal[t+1].values()), 
                                       bounds_error=False, fill_value=(V_optimal[t+1][min_W], V_optimal[t+1][max_W]))
                    # get the expected return
                    expected = p*interp_V(W_next_a) + (1-p)*interp_V(W_next_b) #get expected return
                
                # get the optimal value function
                if expected > best_value: 
                    best_value = expected
            # store the optimal value function
            V_optimal[t][W_t] = best_value

    # generate training data
    states, targets = [], []
    for t in V_optimal:
        for W in V_optimal[t]:
            states.append(normalize(t, W))
            targets.append(V_optimal[t][W])

    # convert the data into tensor
    states = torch.FloatTensor(np.array(states))
    targets = torch.FloatTensor(np.array(targets)).unsqueeze(1)

    # train the model 
    for epoch in range(num_epochs):
        optimizer.zero_grad() 
        outputs = model(states) # get the output of the model
        loss = criterion(outputs, targets) # calculate the loss value
        loss.backward() # backpropagation
        optimizer.step() # update the weights
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}") # print the loss value
    return model

# get the optimal path of investment strategy and wealth by using the trained DNN model
def simulate_optimal_path(model, W0, T, p, A, B, r):
    # prepare the data
    # normalize the wealth to [0, 1] range according to the maximum and minimum wealth
    max_single_return = max(A, B, r)
    min_single_return = min(A, B, r)
    max_possible_W = W0 * (1 + max_single_return) ** T
    min_possible_W = W0 * (1 + min_single_return) ** T
    max_W = max_possible_W
    min_W = min_possible_W
    normalize = lambda t, W: np.array([t / T, (W - min_W) / (max_W - min_W)])
    
    current_W = W0 # set the initial wealth
    optimal_actions = [] # store the optimal investment strategy
    wealth_path = [current_W] # store the wealth path

    # get through the time horizon to get the optimal investment strategy and wealth path
    for t in range(T):
        best_x, best_value = 0.0, -np.inf
        for x_t in np.linspace(0, 1, 11):
            W_next_a = x_t * current_W*(1+A) + (1-x_t)*current_W*(1+r)
            W_next_b = x_t * current_W*(1+B) + (1-x_t)*current_W*(1+r)
            
           # get the expected return of the optimal action
            with torch.no_grad(): 
                # get reward of current action at time t with differnt riksy asset return rate
                V_a = model(torch.FloatTensor(normalize(t+1, W_next_a)).unsqueeze(0)).item()
                V_b = model(torch.FloatTensor(normalize(t+1, W_next_b)).unsqueeze(0)).item()
            expected = p*V_a + (1-p)*V_b # get expected return
            
            # get the optimal action
            if expected > best_value:
                best_value, best_x = expected, x_t
        
        # update the expected return of the optimal action
        E_return = best_x*(p*A + (1-p)*B) + (1-best_x)*r  # get expected return rate of the wealth set
        current_W *= (1 + E_return) # get expected value of the wealth set
        optimal_actions.append(best_x)
        wealth_path.append(current_W)
    
    return optimal_actions, wealth_path

if __name__ == "__main__":
    # set global parameters
    T = 10 # time horizon
    a = 1 # risk aversion coefficient of utility function
    p = 0.6 # probability of risky asset getting positive return
    A = 0.1 # expected return rate of risky asset
    B = -0.1 # expected return rate of risk asset
    r = 0.02 # risk-free return rate
    W0 = np.random.uniform(1,20) # generate an initial wealth by uniform distribution in range [1, 20]

    # train the DNN model and get the optimal investment strategy
    dnn_model = train_DNN(T, a, p, A, B, r,W0, num_epochs=1000)
    actions, wealth = simulate_optimal_path(dnn_model, W0, T, p, A, B, r)

    # print the optimal investment strategy and wealth path
    print("\nOptimal Investment Strategy and Wealth Path:")
    print("Time\tInvestment(%)\tWealth")
    for t in range(T):
        print(f"{t}\t{actions[t]*100:.1f}%\t\t{wealth[t]:.2f}")
    print(f"{T}\t-\t\t{wealth[-1]:.2f}")
