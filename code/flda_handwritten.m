%%FLDA ON THE HANDWRITTEN DATASETS
%%EXTRACT THE DATASETSN 
clear all;
clc;
close all;

%extracting data from the files
data_path = 'DATA\HANDWRITTEN DATA'; 
hand_train=cell(1,5);
hand_develop=cell(1,5);
ctra=zeros(1,5);
ctst=zeros(1,5);
count=1;
for i=3:7
ind_data=1;
a=dir('DATA\HANDWRITTEN DATA');
c=a(i).name;
%tname = strcat(data_path, c, '/train');
f = fullfile(data_path,c,'train');
g=fullfile(data_path,c,'dev');
a = dir(fullfile(f, '*.txt'));
b=dir(fullfile(g, '*.txt'));
tdata=cell(1,length(a));
ddata=cell(1,length(b));
hand_train{count}=tdata;
hand_develop{count}=ddata;
for j=1:length(a)
  tmfc=fullfile(f,a(j).name);
  format long g
 hand_train{count}{j} =dlmread(tmfc);
 end
for j=1:length(b)
  tmfc=fullfile(g,b(j).name);
  format long g
 hand_develop{count}{j} =dlmread(tmfc);
 end  
 ctra(1,count)=length(a);
 ctst(1,count)=length(b);
 count=count+1; 
end
%%
%MAKE ALL THE TRAINING FILE OF THE SAME LENGTH
v=zeros(sum(ctra),1);
idx=1;
hand_train_norm=cell(1,5);
for i=1:5
    for j=1:ctra(1,i)
        v(idx,1)=hand_train{i}{j}(1);
        idx=idx+1;
    end
end
norm_length=max(v);
%%
for i=1:5
    hand_train_norm{i}=hand_train{i}
    for j=1:ctra(1,i)
        hand_train_norm{i}{j}=zeros(2*norm_length,1);
        hand_train_norm{i}{j}=hand_train{i}{j}(2:end);
        z=hand_train{i}{j}(1);
        if hand_train{i}{j}(end)- hand_train{i}{j}(end-2)<0.01
           
             for k=1:2: norm_length-z
                
               hand_train_norm{i}{j}(z+k)=hand_train{i}{j}(end-5)+0.17;
                hand_train{i}{j}(end-1)=hand_train_norm{i}{j}(z+k);
             end
             for k=2:2: norm_length-z
                
               hand_train_norm{i}{j}(z+k)=hand_train{i}{j}(end)
                hand_train{i}{j}(end)=hand_train_norm{i}{j}(z+k);
             end
        end
        if hand_train{i}{j}(end-1)- hand_train{i}{j}(end-3)<0.01
           
             for k=2:2: norm_length-z
                
               hand_train_norm{i}{j}(z+k)=hand_train{i}{j}(end)+0.17
                hand_train{i}{j}(end)=hand_train_norm{i}{j}(z+k);
             end
             for k=1:2: norm_length-z
                
               hand_train_norm{i}{j}(z+k)=hand_train{i}{j}(end-1)
                hand_train{i}{j}(end-1)=hand_train_norm{i}{j}(z+k);
             end
        end
    end
end
%%
%%
padded=zeros(norm_length*6,sum(ctra));
count=1;
for i=1:5
   for j=1:ctra(1,i)
        hand_train{i}{j}=fext(hand_train{i}{j});
        %zero_length=norm_length-hand_train{i}{j}(1,2);
        a'=reshape(hand_train{i}{j}(2:end,:),[],1);
       
        padded(:,count)=b;
        count=count+1;
        
   end
end
[coeff,score,~,~,explained]=pca(padded');
sum_explained=0;
idx=0;
while sum_explained<80
    idx=idx+1;
    sum_explained=sum_explained+explained(idx);
end
%coeff=coeff(:,1:143); %
%%MAKE ALL THE TRAINING FILE OF THE SAME LENGTH
%%NOW APPLY THE FLDA AS APPLIED FOR THE IMAGE DATA
%%MAKE THE CLASSIFICATION USING THE BAYESIAN CLASSIFIER
%%MAKE THE CONFUSION MATRX
