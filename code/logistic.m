function [theta_i]=logistic(X,y)
%X it is the bigfat matrix with each row is the no. of observation 
%y is the true label of the datasets
%since,multiclass calssification each:
%positive had been made as 1 and negative as 0
[m n]=size(X);
intialtheta=zeros((n+1),1);
%no. of parameter that can be used as the function and adding the bias
[j grad]=computecost(intialtheta,X,y);

%finding the optimum value of the weight vector
%hyperparameters and max_iters
alpha=0.001;
iter=500;
theta_i=intialtheta;

%above three values are the hyperparameter
for i=1:iter
    theta_f=theta_i-alpha*grad;
    [j grad]=computecost(theta_f,X,y);
    
        
    
    theta_i=theta_f;
    
end
end