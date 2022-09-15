%%FLDA FOR THE SYNTHETIC DATA
%%
%EXTRACTING THE DATASETS FOR THE SYNTHETIC DATA
clear all;
clc;
close all;

data_path = 'data/Synthetic_Data/';

tdata_ind = 1;
ddata_ind = 1;
tname = strcat(data_path, '/train.txt');
dname = strcat(data_path, '/dev.txt');

fid = fopen(tname,'r');
xx = textscan(fid,'%f,%f,%d');
fclose(fid);
tdata = [xx{1} xx{2}];

tnum = length(xx{1});
%extracting the develop data
fid = fopen(dname,'r');
yy = textscan(fid,'%f,%f,%d');
fclose(fid);
ddata = [yy{1} yy{2}];

dnum = length(yy{1});

%%
%MAKE THE FLDA AS THE SYNTHETIC DATASETS
count_train=zeros(1,2);

for i=1:length(xx{3})
    if xx{3}(i)==1
        count_train(1,1)=count_train(1,1)+1;
    end
    if xx{3}(i)==2
        count_train(1,2)=count_train(1,2)+1;
    end
end
count1=1;


for i=1:1
    A_with=zeros(count_train(i),2);
    for j=1:count_train(i)
        A_with(j,:)=tdata(count1,:);
        count1=count1+1;
    end
    score_norm=normalize(A_with);
end
for i=2:2
    A_with=zeros(count_train(i),2);
    for j=1:count_train(i)
        A_with(j,:)=tdata(count1,:);
        count1=count1+1;
    end
    score_norm=[score_norm ;normalize(A_with)];
end
%%
%making the mean matrix for the use in the case of the 
mn_vec=zeros(2,2);
id=1;
for i=1:2
    sum=0;
    for j=1:count_train(i)
        sum=sum+score_norm(id,:);
        id=id+1;
    end
    mn_vec(i,:)=sum/count_train(i);

end
%%

%make the total scatter matrix
[v,~]=size(score_norm);
m=zeros(v,1);
for i=1:v
    m=m+score_norm(i,:);
end
m_n=m/v;
S_t=zeros(2,2);
for j=1:v
    S_t=S_t+(score_norm(j,:)'-m_n')*(score_norm(j,:)'-m_n')';
end

%make the within class matrix
S_w=zeros(2,2);
 idx=1;  
for i=1:2
    S1=zeros(2,2);
    for j=1:count_train(i)
        S1=score_norm(idx,:)'-mn_vec(i,:)';
        idx=idx+1;
    end
    S_w=S_w+S1*S1';
end
S_b=S_t-S_w;
%finding the direction that can be used in the case of 
[V D]=eig(inv(S_w)*S_b);
%%
m=1;
V_s=zeros(2,m);
 
for i=1:m
  V_s(:,i)=V(:,i);
end
project=tdata*V_s;


%%
%MAKE THE BAYESIAN CLASSIFIER FOR THE CLASSIFICAION
mean=zeros(2,m);
cova=zeros(m,m,2);
count1=1;
for i=1:2
    A_with=zeros(count_train(i),m);
    for j=1:count_train(i)
        A_with(j,:)=project(count1,:);
        
        count1=count1+1;
    end
    [mean(i,:),cova(:,:,i)]=my_norm(A_with);
    
end
%%
count_develop=zeros(1,2);

for i=1:length(yy{3})
    if yy{3}(i)==1
        count_develop(1,1)=count_develop(1,1)+1;
    end
    if yy{3}(i)==2
        count_develop(1,2)=count_develop(1,2)+1;
    end
end

%%
%project the development data on the same axes
dev_ud=ddata*V_s;
predict_data=zeros(dnum,2);
predict=zeros(dnum,1);
prior=count_develop/dnum;
count1=1;
for i=1:2
    
    for j=1:count_develop(i)
        for k=1:2
            predict_data(count1,k)=prior(1,k)*GetNDGaussian(dev_ud(count1,:),mean(k,:),cova(:,:,k));
        end
       predict(count1)=find(predict_data(count1,:)==max(predict_data(count1,:)));
        count1=count1+1;
    end
    
end

%%
%MAKE THE CONFUSION MATRIX
confusion_matrix(double(yy{3})',predict');
%%
%plot the roc curve for the flda synthetic data
targets=zeros(dnum,2);
j=1;
for i=1:dnum
    if yy{3}(i)==1
        targets(i,1)=1;
    end
    if yy{3}(i)==2
        targets(i,2)=1;
    end
end
[tpr fpr thersholds]=roc(targets,predict_data);
plotroc(tpr,fpr);
%%
%plot  det curve for the flda syntheticc class
targ=zeros(1000,1);
nontarg=zeros(1000,1);
for i=1:dnum
    if yy{3}(i)==1
        targ(i,1)=predict_data(i,1);
        nontarg(i,1)=predict_data(i,2);
    end
    if yy{3}(i)==2
        targ(i,1)=predict_data(i,2);
        nontarg(i,1)=predict_data(i,1);
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
   