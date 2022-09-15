%% 
%extraction the daata
clear all;
clc;
close all;

%extracting train and dev image_data
data_path = 'data\SPEECH DATA'; 
sp_train=cell(1,5);
sp_develop=cell(1,5);
ctra=zeros(1,5);
ctst=zeros(1,5);


%finding the directory of the given datasets
%1=coast,%2=highway,%3=insidecity,%4=opencountry,%5=tallbuilding
count=1;
for i=3:7
    dlist=dir('data\SPEECH DATA');
    c=dlist(i).name;
    
    t_file=fullfile(data_path,c,'train');
    d_file=fullfile(data_path,c,'dev');
    
    tdata_list= dir(fullfile(t_file, '*.mfcc'));
    ddata_list=dir(fullfile(d_file, '*.mfcc'));
   
    tdata=cell(1,length(tdata_list));
    ddata=cell(1,length(ddata_list));
     
    im_train{count}=tdata;
    im_develop{count}=ddata;
    
    for j=1:length(tdata_list)
        tmfc=fullfile(t_file,tdata_list(j).name);
        format long g
        im_train{count}{j} =dlmread(tmfc);
    end
    
    for j=1:length(ddata_list)
        tmfc=fullfile(d_file,ddata_list(j).name);
        format long g
        im_develop{count}{j} =dlmread(tmfc);
    end  
count=count+1;

end
%%
%reshaping of the data for the masking of the UBM GMM model
A=im_train{1}{1}(2:end,:);
for i=1:5
    for j=1:length(im_train{i})
        a=im_train{i}{j}(2:end,:);
        A=[A;a];
    end
end
[v u]=size(im_train{1}{1}(2:end,:));
B=A(v+1:end,:);
        

%%
%Build the UBM GMM model\

%%Map adaptation of the data and can be used for each of the training data