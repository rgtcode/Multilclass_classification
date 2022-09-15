function logLikelihood=getLogLikelihood(means,weights,covariances,data)
[N D]=size(data);
k=length(weights);
%finding the loglikelihood for each of the data points
logLikelihood=0;
for i=1:N 
    p=0;
    for j=1:k %for each of the gaussian mixture model
        meansDiff=data(i,:)-means(j,:);
        covariance=covariances(:,:,j);
        norm=1/(((2*pi)^(D/2))*sqrt(det(covariance)));
        p=p+weights(1,j)*norm*(exp(-0.5*((meansDiff*inv(covariance)*meansDiff'))));
    end
    
    logLikelihood=logLikelihood+log(p);
end
end