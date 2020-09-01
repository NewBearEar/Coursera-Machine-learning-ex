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

% 正则化项theta从theta1开始，对应theta向量的第二项
J = -1.0.*sum(y.*log(sigmoid(X*theta))+(1-y).*log(1-sigmoid(X*theta)))./m + lambda.*sum(theta(2:end,1).^2)./(2*m);
% 计算不含正则项的所有grad
grad = 1.0.*(X'*(sigmoid(X*theta)-y))./m;
% 加上正则项的grad
grad(2:end,1) = grad(2:end,1) + lambda.*theta(2:end,1)./m;



% =============================================================

end
