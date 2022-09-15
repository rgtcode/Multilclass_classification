clear all;
clc;
close all;

%extracting data from the files
data_path = 'DATA\HANDWRITTEN DATA'; 
sp_train=cell(1,5);
sp_develop=cell(1,5);
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
 ctra(1,count)=length(a);
 ctst(1,count)=length(b);
 count=count+1; 
end
for i=1:5
   for j=1:ctra(1,i)
        sp_train{i}{j}=fext(sp_train{i}{j});
   end
end
for i=1:5
   for j=1:ctst(1,i)
        sp_develop{i}{j}=fext(sp_develop{i}{j});
   end
end
%%
% For prediction of handwritten characters
 predict=zeros(1,sum(ctst));
actual=zeros(1,sum(ctst));
count1=1;
for g=1:5
    for l=1:ctst(1,g)
        actual(1,count1)=g;
        d=zeros(1,sum(ctra));
        
        count=1;
       for i=1:5
               for j=1:ctra(1,i)
                  c= mydtw2(sp_train{i}{j},sp_develop{g}{l});
                   d(1,count)=c;
                  count=count+1;
                   
               end
         end
count=0;
          e=sort(d,'ascend');
          f=e(1,1:11);
          loc=zeros(1,length(f));
         for k=1:length(f)
             loc(1,k)=find(f(1,k)==d);
             if (1<=loc(1,k)&&loc(1,k)<=ctra(1,1))
                 loc(1,k)=1;
             end
            if (ctra(1,1)+1<=loc(1,k)&&loc(1,k)<=sum(ctra(1:2)))
                 loc(1,k)=2;
             end
            if (sum(ctra(1:2))+1<=loc(1,k)&&loc(1,k)<=sum(ctra(1:3)))
                loc(1,k)=3;
            end
            if (sum(ctra(1:3))+1<=loc(1,k)&&loc(1,k)<=sum(ctra(1:4)))
                loc(1,k)=4;
            end
            if (sum(ctra(1:4))+1<=loc(1,k)&&loc(1,k)<=sum(ctra))
                loc(1,k)=5;
            end
         end
        cout=zeros(1,5);
      
        for i=1:length(loc)
            if (loc(1,i)==1)
                cout(1,1)=cout(1,1)+1;
            end
             if (loc(1,i)==2)
                cout(1,2)=cout(1,2)+1;
             end
             if (loc(1,i)==3)
                 cout(1,3)=cout(1,3)+1;
             end
            if (loc(1,i)==4)
                 cout(1,4)=cout(1,4)+1;
             end
               if (loc(1,i)==5)
                 cout(1,5)=cout(1,5)+1;
             end

         end
   d=find(max(cout)==cout);

          predict(1,count1)=d(1,1);
      count1=count1+1;
    end 
end
%%
%plot the confusion matrix
actual=zeros(1,100);
co=1;
for i=1:5
    for j= 1:ctst(i)
        actual(1,co)=i;
        co=co+1;
    end
end
confusion_matrix(actual,predict)
%%
%plot the roc curve
 dist=zeros(sum(ctst),5);
target=zeros(sum(ctst),5);

for i=1:5
    for k=1:ctst(1,i)
        
        for j=1:5
              x=zeros(1,ctra(1,j));
            for n=1:ctra(1,j)
                c=mydtw2(sp_train{j}{n},sp_develop{i}{k});
                x(1,n)=c;
            end
            dist(k,j)=min(x);
        end
    end
end
co=1;
for i=1:5
    for j=1:ctst(i)
        target(co,i)=1;
        co=co+1;
    end
end
%%
[tpr fpr thersholds]=roc(target,dist);
plotroc(tpr,fpr);
%%
%plot the det curve
targ=zeros(1,100);
nontarg=zeros(1,400);


for n=1:5
    targ(1,(n-1)*20+1:20*n)=dist((n-1)*20+1:20*n,n);
end
nc=1;
ll=1;
for i = 1:5
    for j = 1 : 20
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
                