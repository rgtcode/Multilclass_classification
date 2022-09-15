function [j grad]=computecost(theta,X,y)

%calculating the size of the matrix
[m ~]=size(X);
%adding the bias column to the main coliumn
X=[ones(m,1) X];
t=(X*theta);
%making the sigmoid function
z=1./(1+exp(-t));
j=-(1/m)*sum(y.*log(z)+(1-y).*log(1-z));

%for calculating the gradient of the function
[n,~]=size(theta);
grad=zeros(n,1);

for i=1:n
    grad(i)=((z-y)'*X(:,i));
end
end