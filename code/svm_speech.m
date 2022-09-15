clc;
clear all;
close all;
exam=matfile('D.mat');
TX=exam.X;
et=matfile('countt.mat');
count_train=et.count_train;
%%
T=zeros(1,sum(count_train));
count=1;
for i=1:5
    for j=1:count_train(i)
        T(1,count)=i;
        count=count+1;
    end
end
%%
md1=fitcecoc(TX,T);
exam=matfile('E.mat');
Td=exam.dev;
predict=predict(md1,Td);
%%
ex=matfile('countd.mat');
count_develop=ex.count_develop;
%%
actual=zeros(1,sum(count_develop));
%%
count=1;
for i=1:5
    for j=1:count_develop(i)
        actual(1,count)=i;
        count=count+1;
    end
end
%%
confusion_matrix(actual,predict');

