%%FLDA ON THE SPEECH DATA
%%EXTRACTION OF THE DATASETS
%%clear all;
clc;
clear all;
close all;

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
% %%MAKE THE USE OF THE SAME MATRIX EVERYWHERE i_VECTOR
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
% 
% %%
% 
% sigma=blkdiag(cov(:,:,1),cov(:,:,2),cov(:,:,3),cov(:,:,4),cov(:,:,5),cov(:,:,6),cov(:,:,7),cov(:,:,8),cov(:,:,9),cov(:,:,10),cov(:,:,11),cov(:,:,12),cov(:,:,13),cov(:,:,14),cov(:,:,15),cov(:,:,16),cov(:,:,17),cov(:,:,18),cov(:,:,19),cov(:,:,20));
% sigma1=inv(sigma);
% 
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
% X=zeros(sum(count_train),10);
% count3=1;
% for i=1:5
%     for j=1:count_train{i}
%         X(count3,:)=I(j,:,i);
%         count3=count3+1;
%     end
% end
%    
exam=matfile('D.mat');
TX=exam.X;
%%APPLY THE FLDA TO IT 
%within class covariance normalisation of the score data
count1=1;
ec=matfile('countt.mat');
count_train=ec.count_train;
rank=10;

for i=1:1
    A_with=zeros(count_train(i),rank);
    for j=1:count_train(i)
        A_with(j,:)=TX(count1,:);
        count1=count1+1;
    end
    score_norm=normalize(A_with);
end
for i=2:5
    A_with=zeros(count_train(i),rank);
    for j=1:count_train(i)
        A_with(j,:)=TX(count1,:);
        count1=count1+1;
    end
    score_norm=[score_norm ;normalize(A_with)];
end
%%
% %finding the mean of each class separately
% 
mn_vec=zeros(5,rank);
id=1;
for i=1:5
    sum=0;
    for j=1:count_train(i)
        sum=sum+score_norm(id,:);
        id=id+1;
    end
    mn_vec(i,:)=sum/count_train(i);

end
%%
idx=1;
%finding the total scatter matrix
S_w=zeros(10,10);
[v,~]=size(score_norm);
m=zeros(v,1);
for i=1:v
    m=m+score_norm(i,:);
end
m_n=m/v;
S_t=zeros(rank,rank);
for j=1:v
    S_t=S_t+(score_norm(j,:)'-m_n')*(score_norm(j,:)'-m_n')';
end
idx=1;  
for i=1:5
    S1=zeros(rank,rank);
    for j=1:count_train(i)
        S1=score_norm(idx,:)'-mn_vec(i,:)';
        idx=idx+1;
    end
    S_w=S_w+S1*S1';
end
S_b=S_t-S_w;
%%
epsilon=0.0001;
S_w=S_w+epsilon;
[V D]=eig(inv(S_w)*S_b);
m=4;
V_s=zeros(rank,m);
 
for i=1:m
  V_s(:,i)=V(:,i);
end
project=TX*V_s;
%%
count1=1;
for i=1:5
    A_with=zeros(count_train(i),m);
    for j=1:count_train(i)
        A_with(j,:)=project(count1,:);
        
        count1=count1+1;
    end
    [mean(i,:),cova(:,:,i)]=my_norm(A_with);
    
end
%%
% %gnerate the I_vector agan for the development data
% %calculation of the baum welch statistics for the I-vector
% F_s=zeros(sum(count_develop),k,dim);
% N_s=zeros(sum(count_develop),k);
% co=1;
%  for i=1:5
%      for j=1:count_develop(1,i)
%          [q w]=B_ws(im_develop{i}{j},weights,mn,cov,k);
%          F_s(co,:,:)=w;
%          N_s(co,:)=q;
%          co=co+1;
%      end
%  end
% %rank=10;
% I=zeros(39,rank,5);
% co=1;
% for i=1:5
%      for j=1:count_develop(i)
%         I(j,:,i) =I_vec(im_develop{i}{j},sigma1,F_s(co,:,:),N_s(co,:),F_s,N_s,rank);
%         
% 
%          co=co+1;
%      end
% end
% %%
% dev=zeros(sum(count_develop),rank);
% count3=1;
% for i=1:5
%     for j=1:count_train{i}
%         dev(count3,:)=I(j,:,i);
%         count3=count3+1;
%     end
% end
% dev_ud=dev*V_s;
exam=matfile('E.mat');
dev_ud=exam.dev;
dev_ud=dev_ud*V_s;
ex=matfile('countd.mat');
count_develop=ex.count_develop;
%%
predict_data=zeros(60,5);
predict=zeros(60,1);
prior=count_develop/60;
count1=1;
for i=1:5
    
    for j=1:count_develop(i)
        for k=1:5
            predict_data(count1,k)=prior(1,k)*GetNDGaussian(dev_ud(count1,:),mean(k,:),cova(:,:,k));
        end
       predict(count1)=find(predict_data(count1,:)==max(predict_data(count1,:)));
        count1=count1+1;
    end
    
end
%%MAKE THE CLASSIFICATION BY THE USE OF THE BAYESIAN CLASSIFIER
%%
%MAKE THE CONFUSION MATRIX
%%
%plotting the confusion matrix
%%
ex=matfile('countd.mat');
count_develop=ex.count_develop;
%%
actual=zeros(1,60);
count=1;
for i=1:5
    for j=1:count_develop(i)
        actual(1,count)=i;
        count=count+1;
    end
end
confusion_matrix(actual,predict');
%%
%plot the roc curve for the flda speech
target=zeros(60,5);
co=1;
for i=1:5
    for j=1:count_develop(i)
        target(co,i)=1;
        co=co+1;
    end
end
[tpr fpr thersholds]=roc(target,predict_data);
plotroc(tpr,fpr);
%%
%plot the det for the flda speech
%%
%plot the det plot for the flda imaeg plot
targ=zeros(60,1);
nontrag=zeros(4*60,1);
k=1;
co=1;
ll=1;
for i=1:5
    for j=1:count_develop(i)
        for k=1:5
            if i==k
              targ(co,1)=predict_data(co,i);
            end
            if i~=k
              nontarg(ll,1)=predict_data(co,k);
              ll=ll+1;
            end
        end
    end
end
plot_title = 'DET plot example';
prior = 0.3;

plot_type = Det_Plot.make_plot_window_from_string('old');
plot_obj = Det_Plot(plot_type,plot_title);

plot_obj.set_system(targ',nontarg','hw');
plot_obj.plot_steppy_det({'b','LineWidth',2},' ');
plot_obj.plot_DR30_fa('c--','30 false alarms');
plot_obj.plot_DR30_miss('k--','30 misses');
plot_obj.plot_mindcf_point(prior,{'b*','MarkerSize',8},'mindcf');

 plot_obj.set_system(targ',nontarg','hw123');
plot_obj.plot_steppy_det({'r','LineWidth',2},' ');
plot_obj.plot_DR30_fa('m--','30 false alarms');
plot_obj.plot_DR30_miss('g--','30 misses');
 plot_obj.plot_mindcf_point(prior,{'r*','MarkerSize',8},'mindcf');

plot_obj.display_legend();
                