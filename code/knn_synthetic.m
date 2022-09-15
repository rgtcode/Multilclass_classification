%%
%EXTRACTION OF THE SYNTHETIC DATA
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
%FINDING THE K-NN
k=10;
kn=zeros(dnum,tnum);
kn_sort=zeros(dnum,tnum);
kn_top=zeros(dnum,10);
for j=1:dnum
    for i=1:tnum
        kn(j,i)=norm(ddata(j,:)-tdata(i,:));
    end
    kn_sort(j,:)=sort(kn(j,:),'ascend');
    kn_top(j,:)=kn_sort(j,1:k);
end
%%
%taking the top k values
loc=zeros(dnum,2);
predict=zeros(dnum,1);
for j=1:dnum
    for i=1:k
       a= find(kn_top(j,i)==kn(j,:));
       if a<=1250
           loc(j,1)=loc(j,1)+1;
       end
       if a>1250
           loc(j,2)=loc(j,2)+1;
       end
    end
    predict(j)=find(max(loc(j,:))==(loc(j,:)));
end
%%
%making the confusion matrix
actual=zeros(1000,1);
for i=1:dnum
    if yy{3}(i)==1
        actual(i)=1;
    end
    if yy{3}(i)==2
        actual(i)=2;
    end
end
confusion_matrix(actual',predict')
%%
%Making the roc curve
count_train=zeros(1,2);
for i=1:tnum
    if xx{3}(i)==1
        count_train(1,1)=count_train(1,1)+1;
    end
    if xx{3}(i)==2
        count_train(1,2)=count_train(1,2)+1;
    end
end
%%
x=zeros(dnum,2);
j=1;
for i=1:dnum
    y=zeros(count_train(1,1),1);
    for l=1:count_train(1,1)
        y(l)=norm(ddata(i,:)-tdata(l,:));
    end
    
    x(i,1)=min(y);
    
end
%%
for i=1:dnum
    y=zeros(count_train(1,2),1);
    co=1;
    for l=1250:2500
        y(co)=norm(ddata(i,:)-tdata(l,:));
        co=co+1;
    end
    x(i,2)=min(y);
    
end
%%
count_develop=zeros(1,2);
for i=1:dnum
    if yy{3}(i)==1
        count_develop(1,1)=count_develop(1,1)+1;
    end
    if yy{3}(i)==2
        count_develop(1,2)=count_develop(1,2)+1;
    end
end
T=zeros(dnum,2);
for i=1:count_develop(1,1)
    T(i,1)=1;
end
for i=501:1000
    T(i,2)=1;
end
%%
[tpr fpr thersholds]=roc(T,x);
plotroc(tpr,fpr);
%%
%plotting the det plot
targ=zeros(1000,1);
nontrag=zeros(1000,1);
for i=1:length(x)
    if i<=500
        targ(i,1)=x(i,1);
        nontarg(i,1)=x(i,2);
    end
    if i>500
        targ(i,1)=x(i,2);
        nontarg(i,1)=x(i,1);
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
              
    
    
