%%Extraction of the handwritten data
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
%extraction of the features for the handwritten data
for i=1:5
   for j=1:ctra(1,i)
        hand_train{i}{j}=fext(hand_train{i}{j});
   end
   
end
for i=1:5
   for j=1:ctst(1,i)
        hand_develop{i}{j}=fext(hand_develop{i}{j});
   end
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
%padding zeros to make it to the constant length as other vector
%
padded=zeros(sum(ctra),norm_length*6);
count=1;
for i=1:5
   for j=1:ctra(1,i)
        hand_train{i}{j}=fext(hand_train{i}{j});
        %zero_length=norm_length-hand_train{i}{j}(1,2);
        a=reshape(hand_train{i}{j}(2:end,:),[],1);
        b=a';
        b=[b zeros(1,norm_length-hand{i}{j}(1;
        padded(count,:)=b;
        count=count+1;
        
   end
end