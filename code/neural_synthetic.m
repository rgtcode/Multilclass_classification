%%MAKE THE NEURAL NETWORK FOR  THE SYNTHETIC DATA
%%EXTRACT THE DATA THAT CAN BE USED FOR THE USE IN THE CASE OF NEURAL NET
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
count_train=zeros(1,2);

for i=1:length(xx{3})
    if xx{3}(i)==1
        count_train(1,1)=count_train(1,1)+1;
    end
    if xx{3}(i)==2
        count_train(1,2)=count_train(1,2)+1;
    end
end
T=zeros(2,tnum);
count=1;
for i=1:2
    for j=1:count_train(i)
        T(i,count)=1;
        count=count+1;
    end
end
%%

%train the neural net
net=patternnet(100);
[net tr]=train(net,tdata',T);
%%
D=tdata';
C=ddata';
%%
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
predict_data=zeros(2,dnum);
predict=zeros(dnum,1);
count1=1;
for i=1:2
   for j=1:count_develop(i)
       predict_data(:,count1)=net(C(:,count1));
       predict(count1)=find(predict_data(:,count1)==max(predict_data(:,count1)));
        count1=count1+1;
    end
    
end
%
%Make the confusion matrix for the logistic regression on the synthetic data
confusion_matrix(double(yy{3})',predict');
%%APPLY THE NEURAL NET AS GOING TO BE USED IN THE CASE OF THE IMAGE DATA 
%%MAKE THE PREDICTION LABEL ON THE DEVELOPMENT DATA 
%%MAKE THE CONFUSION MATRIX THAT 