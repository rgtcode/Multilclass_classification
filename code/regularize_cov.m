function regularized_cov=regularize_cov(covariance)
%regularize a covariance matrix by enforcing a minimum
%INPUT:
%covariance:matrix
%epsilon:minimum values of the singular values
epsilon=0.0001;
regularized_cov=covariance + epsilon*eye(size(covariance));

%make sure matrix is symmentric upto 1e-15 decimal
regularized_cov=(regularized_cov+regularized_cov')/2;
end