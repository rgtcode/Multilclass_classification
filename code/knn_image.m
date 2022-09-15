%%
%extracting the daata
clear all;
clc;
close all;

%extracting train and dev image_data
data_path = 'data\image_data'; 
sp_train=cell(1,5);
sp_develop=cell(1,5);
ctra=zeros(1,5);
ctst=zeros(1,5);


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


%%
%reshaping the train and development data

A=reshape(im_train{1}{1}',[],1);
     
for i=1:5
    for j=1:length(im_train{i})
        a=reshape(im_train{i}{j}',[],1);
        A=[A a];
    end
end
A=A(:,2:end)';
B=reshape(im_develop{1}{1}',[],1);
     
for i=1:5
    for j=1:length(im_develop{i})
        a=reshape(im_develop{i}{j}',[],1);
        B=[B a];
    end
end
B=B(:,2:end)';


%%
%find the KNN of the image data
k=10;
[tlen,~]=size(A);
[dlen,~]=size(B);
predict=zeros(1,340);
d_top=zeros(dlen,k);
d=zeros(dlen,tlen);
d_sort=zeros(dlen,tlen);
    
 for j=1:dlen
    
     for l=1:tlen
         d(j,l)= norm(B(j,:)-A(l,:));
         
     end
     d_sort(j,:)=sort(d(j,:),'ascend');
     d_top(j,:)=d_sort(j,1:k);
end
     %%
 loc=zeros(dlen,k);
 predict=zeros(dlen,1);
 a=zeros(1,k);
 for m=1:dlen
     f=zeros(1,5);
     for j=1:k
         a=find(d_top(m,j)==d(m,:));
        
         if (a<=251)
                  f(1,1)=f(1,1)+1;
          end
         if (252<a&&a<=433)
                  f(1,2)=f(1,2)+1;
          end
         if (434<a && a<=648)
                  f(1,3)=f(1,3)+1;
         end
          if (649<a&&a<=935)
                  f(1,4)=f(1,4)+1;
          end
          if (936<a&&a<=1184)
                  f(1,5)=f(1,5)+1;
          end
      end
%    
     t=find(max(f(1,:))==(f(1,:)));
     predict(m)=t(1,1);
 end
%%
%plotting the confusion matrix
%plotting the confusion matrix
actual=zeros(340,1);
co=1;
for i=1:5
    for j=1:length(im_develop{i})
        actual(co,1)=i;
        co=co+1;
    end
end
confusion_matrix(actual',predict');
%%
%plot the roc curve for the given data
tars=zeros(dlen,5);
for j=1:dlen
    l=1;
    for i=1:5
        y=zeros(length(im_train{i}),1);
        for h=1:length(im_train{i})
         y(h)= norm(B(j,:)-A(l,:));
         l=l+1;
        end
     tars(j,i)=min(y);
    end
     
end
target=zeros(340,5);
co=1;
for i=1:5
    for j=1:length(im_develop{i})
        target(co,i)=1;
        co=co+1;
    end
end
[tpr fpr thersholds]=roc(target,tars);
plotroc(tpr,fpr);
%%
%plot the det plot for the flda imaeg plot
targ=zeros(340,1);
nontrag=zeros(4*340,1);
k=1;
co=1;
ll=1;
for i=1:5
    for j=1:length(im_train{i})
        for k=1:5
            if i==k
              targ(co,1)=tars(co,i);
            end
            if i~=k
              nontarg(ll,1)=tars(co,k);
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
                