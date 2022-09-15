function [weights means covariances]=gmmest(data,k,n_iter,covariance,means)

%calculate the dimension of pooled data
% [~,n_dim]=size(data);
% 
% %intialize the weights and covariances
 weights=ones(1,k)/k;
% covariances=zeros(n_dim,n_dim,k);

%use the K-means clustering to intializing the EM algorithm
%%[clusture_idx,means,sumd]=kmeans(data,k,'replicate',5);

%create the intial covariance matrix
% for j=1:k
%     sumd(j,1)=sumd(j,1)./sum(clusture_idx==j);
% end
% for j=1:k
%     covariance(:,:,j)=eye(n_dim).*sumd(j,1);
% end

%Applying the EM algorithm to it
for iters=1:n_iter
    [oldlogi,gamma]=Estep(means,covariance,weights,data);
    [weights,means,covariances,newlogli]=Mstep(gamma,data);
    
    %regularize covariances matrix
    for j=1:k
        covariances(:,:,j)=regularize_cov(covariances(:,:,j));
    end
    
    %termination creation
    if abs(oldlogi-newlogli)<1
        break;
    end
end
end
  
