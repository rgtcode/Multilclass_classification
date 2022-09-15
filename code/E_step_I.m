function [Ls Ws]=E_step_I(T,sig,Fu,Nu,rank)

%first finding Ls that can be used in the maximisation step
%finding the Ws that also can be used in the maximisation step
epsi=0.001;
%regu_sig=sig+epsi;
Ls=eye(rank)+T'*inv(sig)*Nu*T;
%regu_Ls=Ls+epsi;
Ws=inv(Ls+epsi)*T'*inv(sig)*Fu;
end