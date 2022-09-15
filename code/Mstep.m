function [weights,means,covariances,logLikelihood]=Mstep(gamma,data)
%get the size
[nTrainingSamples,dim]=size(data);
K=size(gamma,2);

noloops=0;
if noloops
    
    %update the weights
    N_hat=sum(gamma,1);
    weights=N_hat/nTrainingSamples;
    
    %update the means
    means=bsxfun(@rdivide,gamma'*data,N_hat');
    
    %update the covariances
    covariances=zeros(dim,dim,k);
    for j=1:K
        debiased=data-repmat(means(j,:),nTrainingSamples,1);
        covariances(:,:,j)=(bsxfun(@times.debiased,gamma(:,j))'*debiased)/N_hat(j);
    end
else
    %or use of loops
    %create matrices
    means=zeros(K,dim);
    covariances=zeros(dim,dim,K);
    
    %compute the weights
    Nk=sum(gamma,1);
    weights=Nk./nTrainingSamples;
    for i=1:K
        auxMean=zeros(1,dim);
        for j=1:nTrainingSamples
            auxMean=auxMean+gamma(j,i).*data(j,:);
        end
        means(i,:)=auxMean./Nk(i);
    end
    for i=1:K
        auxSigma=zeros(dim,dim);
        for j=1:nTrainingSamples
            meansDiff=data(j,:)-means(i,:);
            auxSigma=auxSigma+gamma(j,i).*((meansDiff)'*meansDiff);
        end
        covariances(:,:,i)=auxSigma./Nk(i);
    end
end
%compute logLikelihood
logLikelihood=getLogLikelihood(means,weights,covariances,data);
end