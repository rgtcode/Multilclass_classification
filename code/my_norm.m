%%pca on the data
%%finding the orhongonal basis vectors for the each dimensions
function [mean C]=my_norm(data)

%take the data where:- 
%no.of rows is equal to the no. of observation
%no. of columns is the dimension of the matrix
[m n]=size(data);

mn=zeros(1,n);
for i=1:m
    mn(1,:)=mn(1,:)+data(i,:);
end
mean=mn/m;
C=cov(data);

%now data has become the mean normalized
% data_m=data-mean;
% C=0;
% for i=1:m
%     
%         C=C+(data(i,:)-mean)*(data(i,:)-mean)';
%     
% end
% data_norm=C^(-1/2)*data_m;
% %[q v]=eig(covariance);
% %v=diag(v);
end


