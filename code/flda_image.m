
clear all;
clc;
close all;

%extracting train and dev image_data
data_path = 'data\image_data'; 
sp_train=cell(1,5);
sp_develop=cell(1,5);

%finding the directory of the given datasets
%1=coast,%2=highway,%3=insidecity,%4=opencountry,%5=tallbuilding
count=1;
count_train=zeros(1,5);
count_develop=zeros(1,5);


for i=3:7
    dlist=dir('data\image_data'); 
    c=dlist(i).name;
    
    t_file=fullfile(data_path,c,'train');
    d_file=fullfile(data_path,c,'dev');
    
    tdata_list= dir(fullfile(t_file, '*.jpg_color_edh_entropy'));
    ddata_list=dir(fullfile(d_file, '*.jpg_color_edh_entropy'));
   
    tdata=cell(1,length(tdata_list));
    ddata=cell(1,length(ddata_list));
     
    im_train{count}=tdata;
     im_develop{count}=ddata;
    
    for j=1:length(tdata_list)
        tmfc=fullfile(t_file,tdata_list(j).name);
        format long g
        im_train{count}{j} =dlmread(tmfc);
        count_train(1,i-2)=count_train(1,i-2)+1;
    end
    
    for j=1:length(ddata_list)
        tmfc=fullfile(d_file,ddata_list(j).name);
        format long g
        im_develop{count}{j} =dlmread(tmfc);
        count_develop(1,i-2)=count_develop(1,i-2)+1;
    end  
count=count+1;

end

%%
%applying the pca to the image data

%len_tot=length(im_train{1})+length(im_train{2})+length(im_train{3})+length(im_train{4})+length(im_train{5});

A=reshape(im_train{1}{1}',[],1);
     
for i=1:5
    for j=1:length(im_train{i})
        a=reshape(im_train{i}{j}',[],1);
        A=[A a];
    end
end
A=A(:,2:end)';
B=A';
%%

%%
%Now doing the within class covariance normalisation

%%
%A is the matrix having all the 36X23 matrix as the column vectors
[coeff,score,~,~,explained]=pca(A);
sum_explained=0;
idx=0;
while sum_explained<80
    idx=idx+1;
    sum_explained=sum_explained+explained(idx);
end
coeff=coeff(:,1:143); %taking only the top 143 dimensions having 90% of the variances
%%
%within class covariance normalisation of the score data
count1=1;


for i=1:1
    A_with=zeros(count_train(i),828);
    for j=1:count_train(i)
        A_with(j,:)=score(count1,:);
        count1=count1+1;
    end
    score_norm=normalize(A_with);
end
for i=2:5
    A_with=zeros(count_train(i),828);
    for j=1:count_train(i)
        A_with(j,:)=score(count1,:);
        count1=count1+1;
    end
    score_norm=[score_norm ;normalize(A_with)];
end
%%
% %finding the mean of each class separately
% 
mn_vec=zeros(5,828);
id=1;
for i=1:5
    sum=0;
    for j=1:length(im_train{i})
        sum=sum+score_norm(id,:);
        id=id+1;
    end
    mn_vec(i,:)=sum/length(im_train{i});

end
%%
% 
% 
% %within class matrix for the classes
S_w=zeros(828,828);
idx=1;
%finding the total scatter matrix
[v,~]=size(score_norm);
m=zeros(v,1);
for i=1:v
    m=m+score_norm(i,:);
end
m_n=m/v;
S_t=zeros(828,828);
for j=1:v
    S_t=S_t+(score_norm(j,:)'-m_n')*(score_norm(j,:)'-m_n')';
end
%%
 idx=1;  
for i=1:5
    S1=zeros(828,828);
    for j=1:length(im_train{i})
        S1=score_norm(idx,:)'-mn_vec(i,:)';
        idx=idx+1;
    end
    S_w=S_w+S1*S1';
end
%%
% %computing the between class matrix
% mn=zeros(1,828);
% for i=1:5
%     mn=mn+mn_vec(i,:)*length(im_train{i});
% end
% mn=mn/1184;
% 
% 
% for i=1:1184
%     S1=A(i,:)-mn(1,:);
% end
% S_t=S1'*S1;

S_b=S_t-S_w;
%adding the hyperparameter to the S_w becasue it is singular
epsilon=0.0001;
S_w=S_w+epsilon;
[V D]=eig(inv(S_w)*S_b);
m=10;
V_s=zeros(828,m);
 
for i=1:m
  V_s(:,i)=V(:,i);
end
project=A*V_s;
%%
% 
% %finding the S_w and S_b after multiplication with Pc
% 
%S_wn=coeff'*S_w*coeff;
%S_bn=coeff'*S_b*coeff;
% %%
% %Now since, the no.of training sample is comparable to dimension
% %applying the direct LDA solution to it
% [V D]=eig(S_b);
% m=20;
% Y=zeros(828,m);
% for i=1:m
%     Y(:,i)=V(:,i);
% end
% %making the submatrix of the prinicipal submatrix
% D_b=Y'*S_b*Y;
% %doing something equivalent to the whitening transform
% Z=Y*(D_b)^(-1/2);
% ag=Z'*S_b*Z;
% %Diagonalizing the within class covariance matrix
% [U M]=eig(Z'*S_w*Z);
% D_w=U'*Z'*S_w*Z*U;
%%
%finding the mean and covariance of the gaussian for posterior probability
mean=zeros(5,m);
cova=zeros(m,m,5);
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
%for tuning of the parameter and for finding the acurracy of the class
dev_u=reshape(im_develop{1}{1}',[],1);
     
for i=1:5
    for j=1:length(im_develop{i})
        a=reshape(im_develop{i}{j}',[],1);
        dev_u=[dev_u a];
    end
end
dev_u=dev_u(:,2:end)';
%%dev_uu=dev_u*V_s;
dev_ud=dev_u*V_s;
%%
%for rediction of the result that can be used for the 
predict_data=zeros(340,5);
predict=zeros(340,1);
prior=count_develop/340;
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
%%
%plotting the confusion matrix
actual=zeros(340,1);
co=1;
for i=1:5
    for j=1:count_develop(i)
        actual(co,1)=i;
        co=co+1;
    end
end
confusion_matrix(actual',predict');
%%
%plot the roc curve for the flda image
target=zeros(340,5);
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
%plot the det plot for the flda imaeg plot
targ=zeros(340,1);
nontrag=zeros(4*340,1);
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
                