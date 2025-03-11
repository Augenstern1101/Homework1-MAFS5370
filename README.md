# Homework1-MAFS5370
Homework1 of MAFS5370 by CHEN ZILING and CHEN YUWEI.  
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

Our homework is completed by CHEN ZILING and CHEN YUWEI. We worked together on the main code, which means the file hw1.py. While CHEN YUWEI contributes to the unit test and integration test part, and CHEN ZILING contributes to the code comments.

The figure is about our code's output.  
<img width="259" alt="e7726ecff9f44a07d1140ed37d5633b" src="https://github.com/user-attachments/assets/1b020404-dd3f-490a-a2aa-76b635919a00" />
