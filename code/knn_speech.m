clear all;
clc;
close all;

%extracting data from the files
data_path = 'DATA\SPEECH DATA'; 
sp_train=cell(1,5);
sp_develop=cell(1,5);

count=1;
for i=3:7
ind_data=1;
a=dir('DATA\SPEECH DATA');
c=a(i).name;


f = fullfile(data_path,c,'train');
g=fullfile(data_path,c,'dev');


a = dir(fullfile(f, '*.mfcc'));
b=dir(fullfile(g, '*.mfcc'));

tdata=cell(1,length(a));
ddata=cell(1,length(b));
sp_train{count}=tdata;
sp_develop{count}=ddata;
for j=1:length(a)
  tmfc=fullfile(f,a(j).name);
  format long g
 sp_train{count}{j} =dlmread(tmfc);
 end
for j=1:length(b)
  tmfc=fullfile(g,b(j).name);
  format long g
 sp_develop{count}{j} =dlmread(tmfc);
 end  
count=count+1;    
end
%%
%for prediction of isolated digits
dis=zeros(39,39,5);
actual=zeros(1,60);
predict=zeros(1,60);
for n=1:5
    for l=1:12
        actual(1,(n-1)*12+l)=n;
        d=zeros(1,195);
        for m=1:5
            for j=1:39
                c= mydtw2(sp_train{m}{j},sp_develop{n}{l});
                d(1,(m-1)*39+j)=c;
            end
        end
        e=sort(d,'ascend');
        f=e(1,1:10);
        loc=zeros(1,length(f));
        for k=1:length(f)
            loc(1,k)=find(f(1,k)==d);
            if (loc(1,k)<=39)
                loc(1,k)=1;
            end
            if (loc(1,k)>39)
                loc(1,k)=(loc(1,k)-rem(loc(1,k),39))/39+1;
            end
        end
        count=zeros(1,5);
        for i=1:length(loc)
            if (loc(1,i)==1)
                count(1,1)=count(1,1)+1;
            end
             if (loc(1,i)==2)
                count(1,2)=count(1,2)+1;
             end
             if (loc(1,i)==3)
                 count(1,3)=count(1,3)+1;
             end
             if (loc(1,i)==4)
                count(1,4)=count(1,4)+1;
             end
            if (loc(1,i)==5)
                count(1,5)=count(1,5)+1;
            end

         end
  d=find(max(count)==count);
  predict(1,(n-1)*12+l)=d(1,1);
     end 
end
%%
%plot the confusion matrix
actual=zeros(1,60);
count=1;
for i=1:5
    for j=1:12
        actual(1,count)=i;
        count=count+1;
    end
end
confusion_matrix(actual,predict);
%%
%plot the roc plot
%%
%plot the roc curve for the flda speech
target=zeros(60,5);
co=1;
for i=1:5
    for j=1:12
        target(co,i)=1;
        co=co+1;
    end
end
x=zeros(1,39);
dist=zeros(60,5);
for n=1:5
    for l=1:12
        actual(1,(n-1)*12+l)=n;
        d=zeros(1,195);
        for m=1:5
            for j=1:39
                c= mydtw2(sp_train{m}{j},sp_develop{n}{l});
                x(1,j)=c;
             end
         i=find(min(x)==x);
         dist((n-1)*12+l,m)=min(x);
       end
    end
end
[tpr fpr thersholds]=roc(target,dist);
plotroc(tpr,fpr);
%%
%plot the det curve
targ=zeros(1,60);
for n=1:5
    targ(1,(n-1)*12+1:12*n)=dist((n-1)*12+1:12*n,n);
end
nontarg=zeros(1,240);
nc =1;
ll = 1;
for i = 1:5
    for j = 1 : 12
        t = 1;
        for k = 1:5
            if i~=k
                nontarg(1,nc) = dist(ll,k);
                nc = nc+1;
                t = t + 1;
            end
        end
        ll = ll+1;
    end
end

plot_title = 'DET plot example';
prior = 0.3;

plot_type = Det_Plot.make_plot_window_from_string('old');
plot_obj = Det_Plot(plot_type,plot_title);

plot_obj.set_system(targ,nontarg,'hw');
plot_obj.plot_steppy_det({'b','LineWidth',2},' ');
plot_obj.plot_DR30_fa('c--','30 false alarms');
plot_obj.plot_DR30_miss('k--','30 misses');
plot_obj.plot_mindcf_point(prior,{'b*','MarkerSize',8},'mindcf');

 plot_obj.set_system(targ,nontarg,'hw123');
plot_obj.plot_steppy_det({'r','LineWidth',2},' ');
plot_obj.plot_DR30_fa('m--','30 false alarms');
plot_obj.plot_DR30_miss('g--','30 misses');
 plot_obj.plot_mindcf_point(prior,{'r*','MarkerSize',8},'mindcf');

plot_obj.display_legend();