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
xx{3}(1501:end)=0;
%%
   
%logisitic regression
y=logist(ddata,double(xx{3}));
%disp(y)
t=zeros(length(yy{1}),1);
for j=1:length(xx{1})
    if(y(j,1)>0.5)
        t(j,1)=1;
    else
        t(j,1)=0;
    end
end


