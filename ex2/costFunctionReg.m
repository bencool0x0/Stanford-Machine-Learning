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

for i = 1:m
  z = theta' * X(i,:)';
  g = sigmoid(z);
  J = J + (-1 * y(i,:) * log(g)) - ((1-y(i,:)) * log(1-g));
end

J = J / (m);

J2 = 0;

for i = 2 : size(theta, 1)
  J2 = J2 + theta(i,:).^2;
end

J2 = J2 * (lambda / (2 * m));

J = J + J2;

temps = zeros(size(X,2), 1);
  for i = 1:size(X,2)
    diff = 0;
    for j = 1:m
      z = X(j,:) * theta;
      g = sigmoid(z);
      %diff = diff + (theta' * X(j,:)' - y(j,:)) * X(j,i);
      diff = diff + (g - y(j,:)) * X(j,i);
    end
    if i > 1
      diff = diff + lambda * theta(i,:);
    endif
    temps(i,1) = ((1 / m) * diff);
    %this is the difference, not the new values of theta
  end
 grad = temps;




% =============================================================

end
