function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% Add bias term to X
X = [ones(m,1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% From here on the digit attached to a variable name can be thought of as
% being a bracket superscript as in Andrew N's notation

% dimensions of X = mxn+1 = mx401 in this case
% dimensions of Theta1 = (number of units in next layer) x n+1
% dimensions of Theta1 = 25x401 

z1 = X * Theta1'; % mx401 * (25x401)' = mx25

% Applying the activation function to each input
a2 = sigmoid(z1);

% add the bias term of 1's to the units in the second layer
% this makes a2 mx26
a2 = [ones(m,1) a2];

% dimensions of Theta2 = number of units in the next layer
%						x number of units in previous layer + 1 (bias added)
% Hence, Theta2 = 10x26
z2 = a2 * Theta2'; % mx26 * (10x26)' = mx10

% applying the activation function to each input in this layer
% The hyp (hypothesis) corresponds to the probability of each class
hyp = sigmoid(z2);

% Assigned class is that with the maximum probability across the rows
% i.e. the index of the maximum probability
% max function returns both the maximum value and its index.
% max variable below stores the maximum value and p stores the index 
% of the maximum value.
[max, p] = max(hyp, [],2);

% =========================================================================


end
