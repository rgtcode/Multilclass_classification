exam=matfile('knnsynfpr.mat');
synfpr=exam.fpr;
exam=matfile('knnsyntpr.mat');
syntpr=exam.tpr;
exam=matfile('flda_synth_fpr.mat');
fldfpr=exam.fpr;
exam=matfile('flda_synth_tpr.mat');
fldtpr=exam.tpr;
exam=matfile('logist_synthfpr.mat');
logfpr=exam.fpr;
exam=matfile('logistisynthe_tpr.mat');
logtpr=exam.tpr;
plotroc(syntpr,synfpr)
hold on
plotroc(fldtpr,fldfpr)
hold on
plotroc(logtpr,logfpr)