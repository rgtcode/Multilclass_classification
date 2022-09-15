function pi_k=probmixgauss(data,k,means)
%k is the no. of the gaussian mixture models
%data=datamatrix
%probability of the each of the gaussian componnents
pi_k=zeros(k,1);

[train_file,~]=size(data);
for i=1:train_file
    dist=zeros(1,k);
    for q=1:k
        x=train_file(j,:)-means(q,:);
        dist(1,q)=norm(x);
    end
    [mn,idx]=min(dist);
    pi_k(1,idx)=pi_k(1,idx)+1;
end
pi_k=pi_k/train_file;
end

    