function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% hypothesis of logistic regression.

hyp = sigmoid(X*theta); 

% in calculating J we exclude regularizing theta1 (which is theta0 in eqn): theta(2:end)
J = -(1/m) * (y'*log(hyp) + ((1-y)' * log(1-hyp)))... % (mx1)'*(mx1) - (mx1)'*(mx1)
     + ((lambda/(2*m)) * (theta(2:end)'*theta(2:end))); % ((n-1)x1)'* (n-1)x1

% gradient of theta1 (or theta0 in eqn). remember theta0 is never regularized
grad(1) = (1/m) * (hyp - y)' * X(:,1); % (mx1)' * mx1

% gradients of all regularized thetas (from 2nd to last theta).
grad(2:end) = (1/m) * X(:,2:end)'*(hyp-y) + (lambda/m)*theta(2:end); % (mxn)' * mx1 + 

% =============================================================

end
