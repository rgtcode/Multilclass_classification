function [F_ts N_ts]= B_ws(utter,weights,means,covariances,k)
%for calculating the baum welch statistics
[frame_length,dim]=size(utter);
evi=zeros(frame_length,k);
for l=1:frame_length
    for j=1:k
        y=GetNDGaussian(utter(l,:),means(j,:),covariances(:,:,j));
        evi(l,j)=weights(1,k)*y;
    end
end
F_ts=zeros(k,dim);
N_ts=zeros(1,k);
sum_evi=zeros(frame_length,1);
for j=1:frame_length
    sum_evi(j)=sum(evi(j,:));
end
% a=zeros(1,k);
% for i=1:k
%      b=weights(1,i)*GetNDGaussian(utter(1,:),means(i,:),covariances(:,:,i));
%      a(1,i)=b/sum_evi(1);
% end
    
for i=1:k
    
    for j=1:frame_length
        a=weights(1,i)*GetNDGaussian(utter(j,:),means(i,:),covariances(:,:,i));
        F_ts(i,:)=F_ts(i,:)+(a/(sum_evi(j))*(utter(j,:)-means(i,:))); 
        N_ts(1,i)=N_ts(1,i)+a/(sum_evi(j));
    end
end
N_s=N_ts(1,1)*ones(1,dim);
for j=2:k
    a=N_ts(1,j)*ones(1,dim);
    N_s=[N_s a];
end
N_s=diag(N_s);

end

    