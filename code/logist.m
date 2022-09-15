function y=logist(X,Y)

[m n]=size(X);
t=zeros(m,1);
%intialising the theta
theta=zeros(n+1,1);
%adding the bias to the training samples
X=[ones(m,1) X];
learn_para=0.001;
for j=1:400
%getting the vector from the sigmoid function

h=zeros(m,1);
h=sigmoid(X*theta);

%computing the cost
%j=-(1/m)*sum(Y.*log(h)+(1-Y).*log(1-h));

grad=zeros(size(theta,1),1);
for i=1:size(grad)
    grad(i)=(1/m)*sum((h-Y)'*X(:,i));
end

theta=theta-learn_para*grad(i);
end

y=sigmoid(X*theta);
% for j=1:m
%     if(y>0.5)
%         t(i,1)=1;
%     else
%         t(i,1)=0;
%     end
end


        







