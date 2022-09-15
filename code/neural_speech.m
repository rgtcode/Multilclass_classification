%%NEURLA NET FOR THE SPEECH DATA 
%%
% %LOGISTIC REGRESSION ON THE SPEECH DATA
% %making the frames from each class and from each data stack together
 clear all;
 clc;
 close all;
% 
% %extracting train and dev image_data
% data_path = 'data\SPEECH DATA'; 
% sp_train=cell(1,5);
% sp_develop=cell(1,5);
% ctra=zeros(1,5);
% ctst=zeros(1,5);
% 
% 
% %finding the directory of the given datasets
% %1=coast,%2=highway,%3=insidecity,%4=opencountry,%5=tallbuilding
% count=1;
% count_train=zeros(1,5);
% count_develop=zeros(1,5);
% for i=3:7
%     dlist=dir('data\SPEECH DATA');
%     c=dlist(i).name;
%     
%     t_file=fullfile(data_path,c,'train');
%     d_file=fullfile(data_path,c,'dev');
%     
%     tdata_list= dir(fullfile(t_file, '*.mfcc'));
%     ddata_list=dir(fullfile(d_file, '*.mfcc'));
%    
%     tdata=cell(1,length(tdata_list));
%     ddata=cell(1,length(ddata_list));
%      
%     im_train{count}=tdata;
%     im_develop{count}=ddata;
%     
%     for j=1:length(tdata_list)
%         tmfc=fullfile(t_file,tdata_list(j).name);
%         format long g
%         im_train{count}{j} =dlmread(tmfc);
%         count_train(1,i-2)=count_train(1,i-2)+1;
%     end
%     
%     for j=1:length(ddata_list)
%         tmfc=fullfile(d_file,ddata_list(j).name);
%         format long g
%         im_develop{count}{j} =dlmread(tmfc);
%         count_develop(1,i-2)=count_develop(1,i-2)+1;
%     end  
% count=count+1;
% 
% end
% %%
% %%
% %reshaping of the data for the making of the UBM GMM model
% A=im_train{1}{1}(2:end,:);
% for i=1:5
%     for j=1:length(im_train{i})
%         a=im_train{i}{j}(2:end,:);
%         A=[A;a];
%     end
% end
% [v u]=size(im_train{1}{1}(2:end,:));
% B=A(v+1:end,:);
% [~,dim]=size(B);
% %%
% [clusture_idx,means,sumd]=kmeans(B,20,'replicate',5);
% %%
% k=20;
% [~,n_dim]=size(B);
% weights=ones(1,k)/k;
% covariances=zeros(n_dim,n_dim,k);
% for j=1:k
%     sumd(j,1)=sumd(j,1)./sum(clusture_idx==j);
% end
% for j=1:k
%     covariance(:,:,j)=eye(n_dim).*sumd(j,1);
% end
% %%
% %making of the GMM supervector by clustering all the frames
% epsilon=0.001;
% k=20;
% [weights,mn,cov]=gmmest(B,k,3,covariance,means); 
% %%
% 
% sigma=blkdiag(cov(:,:,1),cov(:,:,2),cov(:,:,3),cov(:,:,4),cov(:,:,5),cov(:,:,6),cov(:,:,7),cov(:,:,8),cov(:,:,9),cov(:,:,10),cov(:,:,11),cov(:,:,12),cov(:,:,13),cov(:,:,14),cov(:,:,15),cov(:,:,16),cov(:,:,17),cov(:,:,18),cov(:,:,19),cov(:,:,20));
% sigma1=inv(sigma);
% %%
% %calculation of the baum welch statistics for the I-vector
% F_s=zeros(sum(count_train),k,dim);
% N_s=zeros(sum(count_train),k);
% co=1;
%  for i=1:5
%      for j=1:count_train(1,i)
%          [q w]=B_ws(im_train{i}{j},weights,mn,cov,k);
%          F_s(co,:,:)=w;
%          N_s(co,:)=q;
%          co=co+1;
%      end
%  end
%  %%
% %for the calculation of the I-vector
% rank=10;
% I=zeros(39,rank,5);
% co=1;
% for i=1:5
%      for j=1:count_train(i)
%         I(j,:,i) =I_vec(im_train{i}{j},sigma1,F_s(co,:,:),N_s(co,:),F_s,N_s,rank);
%         
% 
%          co=co+1;
%      end
% end
% %%
% X=zeros(sum(count_train),rank);
% count3=1;
% for i=1:5
%     for j=1:count_train{i}
%         X(count3,:)=I(j,:,i);
%         count3=count3+1;
%     end
% end
% A=X;
exam=matfile('D.mat');
TX=exam.X;
ec=matfile('countt.mat');
count_train=ec.count_train;
%%
T=zeros(5,sum(count_train));
count=1;
for i=1:5
    for j=1:count_train(i)
        T(i,count)=1;
        count=count+1;
    end
end
%%
%train the neural net
net=patternnet(150);
[net tr]=train(net,TX',T);
%%
%gnerate the I_vector agan for the development data
%calculation of the baum welch statistics for the I-vector
F_s=zeros(sum(count_develop),k,dim);
N_s=zeros(sum(count_develop),k);
co=1;
 for i=1:5
     for j=1:count_develop(1,i)
         [q w]=B_ws(im_develop{i}{j},weights,mn,cov,k);
         F_s(co,:,:)=w;
         N_s(co,:)=q;
         co=co+1;
     end
 end
 %%
%rank=10;
I=zeros(39,rank,5);
co=1;
for i=1:5
     for j=1:count_develop(i)
        I(j,:,i) =I_vec(im_develop{i}{j},sigma1,F_s(co,:,:),N_s(co,:),F_s,N_s,rank);
        

         co=co+1;
     end
end
%%
ex=matfile('countd.mat');
count_develop=ex.count_develop;
%%
% dev=zeros(sum(count_develop),rank);
% count3=1;
% for i=1:5
%     for j=1:count_train{i}
%         dev(count3,:)=I(j,:,i);
%         count3=count3+1;
%     end
% end
exam=matfile('E.mat');
dev=exam.dev;
C=dev';
%%
predict_data=zeros(5,sum(count_develop));
predict=zeros(sum(count_develop),1);
count1=1;
%idx=1;
for i=1:5
    for j=1:count_develop(i)
    predict_data(:,count1)=net(C(:,count1));
        
       predict(count1)=find(predict_data(:,count1)==max(predict_data(:,count1)));
        count1=count1+1;
        
%idx=idx+1;
    end
    
end
%%plotting the confusion matrix
%%
actual=zeros(1,sum(count_develop));
count=1;
for i=1:5
    for j=1:count_develop(i)
        actual(1,count)=i;
        count=count+1;
    end
end
confusion_matrix(actual,predict');

%%MAKE THE I-VECTOR FOR THE USE IN THE CASE OF THE NEURAL 
%%MAKE THE PREDICT LABEL ON THE DEVELOPMENT DATA
%%MAKE THE CONFUSION MATRIX
