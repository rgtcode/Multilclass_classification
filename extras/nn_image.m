%extraction of the image data
clc;
close all;

%extracting train and dev image_data
data_path = 'data\image_data'; 
sp_train=cell(1,5);
sp_develop=cell(1,5);

%finding the directory of the given datasets
%1=coast,%2=highway,%3=insidecity,%4=opencountry,%5=tallbuilding
count=1;
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
    end
    
    for j=1:length(ddata_list)
        tmfc=fullfile(d_file,ddata_list(j).name);
        format long g
        im_develop{count}{j} =dlmread(tmfc);
    end  
count=count+1;

end

%reshaping of the image data
%%
A=reshape(im_train{1}{1}',[],1);
     
for i=1:5
    for j=1:length(im_train{i})
        a=reshape(im_train{i}{j}',[],1);
        A=[A a];
    end
end
A=A(:,2:end)';
B=A';

%%making the file supportive for the train neural net
%%
T=zeros(5,1184);
count=1;
for i=1:5
    for j=1:length(im_train{i})
        T(i,count)=1;
        count=count+1;
    end
end
%%
B=mnrfit(A,T');
%%
%train the neural net
net=patternnet(100);
[net tr]=train(net,A',T);

%%
%make the developement data as the use for the test case
D=reshape(im_develop{1}{1}',[],1);
     
for i=1:5
    for j=1:length(im_develop{i})
        a=reshape(im_develop{i}{j}',[],1);
        D=[D a];
    end
end
D=D(:,2:end)';
C=D';

%%
%test the neural net
testA=C(:,tr.testInd);
testT=T(:,tr.testInd);
testY=net(testA);
testIndices=vec2ind(testY);
%%