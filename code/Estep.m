function [loglikelihood,gamma]=Estep(means,covariances,weights,data)
%finding the loglikelihood of the data 
loglikelihood=getLogLikelihood(means,weights,covariances,data);
[n_train,dim]=size(data);
k=size(weights,2);
gamma=zeros(n_train,k);

for i=1:n_train
    for j=1:k
        meansDiff=data(i,:)-means(j,:);
        covariance=covariances(:,:,j);
        norm=1/((2*pi)^(dim/2)*sqrt(det(covariance)));
        gamma(i,j)=weights(1,j)*norm*exp(-0.5*((meansDiff/covariance)*meansDiff'));
        gamma(i,j)=gamma(i,j);
    end
   % a=(sum(gamma(i,:))+0.0001);
    gamma(i,:)=gamma(i,:)./sum(gamma(i,:));
end
end
        
        
        
        
        
        
        
        
        
        