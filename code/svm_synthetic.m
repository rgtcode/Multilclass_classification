%%FLDA FOR THE SYNTHETIC DATA
%%
%EXTRACTING THE DATASETS FOR THE SYNTHETIC DATA
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
%extracting the develop data
fid = fopen(dname,'r');
yy = textscan(fid,'%f,%f,%d');
fclose(fid);
ddata = [yy{1} yy{2}];

dnum = length(yy{1});
%%
%making the svm classifier
md1=fitcecoc(tdata,xx{3}');
predict=predict(md1,ddata);
%%
%MAKE THE CONFUSION MATRIX
confusion_matrix(double(yy{3})',double(predict)');