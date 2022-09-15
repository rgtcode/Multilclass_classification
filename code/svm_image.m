%%
%EXTRACT THE DATA THAT CAN BE USED 
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
A=reshape(im_train{1}{1}',[],1);
     
for i=1:5
    for j=1:length(im_train{i})
        a=reshape(im_train{i}{j}',[],1);
        A=[A a];
    end
end
A=A(:,2:end)';
 %%
 tar=zeros(1184,1);
 idx=1;
 for i=1:5
     for j=1:length(im_train{i})
         tar(idx)=i;
         idx=idx+1;
     end
 end
 %%
 %training of the image data
 md1=fitcecoc(A,tar');
 %%
 %% 
%for tuning of the parameter
dev_u=reshape(im_develop{1}{1}',[],1);
     
for i=1:5
    for j=1:length(im_develop{i})
        a=reshape(im_develop{i}{j}',[],1);
        dev_u=[dev_u a];
    end
end
dev_u=dev_u(:,2:end)';
%%
predict=predict(md1,dev_u);
%%
actual=zeros(340,1);
co=1;
for i=1:5
    for j=1:length(im_develop{i})
        actual(co)=i;
        co=co+1;
    end
end
        
   
%%
%MAKE THE CONFUSION MATRIX
confusion_matrix(actual',double(predict)');