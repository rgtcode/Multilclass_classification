function [regu_Ls,Ws]=M_step1(T,sig,Fu,Nu,rank)

%first finding Ls that can be used in the maximisation step
%finding the Ws that also can be used in the maximisation step
epsi=0.001;
regu_sig=sig+epsi;
Ls=eye(rank)+T'*inv(regu_sig)*Nu*T;
regu_Ls=Ls+epsi;
Ws=inv(regu_Ls)*T'*inv(regu_sig)*Fu;
end