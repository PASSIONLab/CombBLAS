% computing the value of the constraint
% I(X;sqrt(eta*mu*snr)X+N)+e*mu/2(eta-1-log(eta))

function y = constraint(e,mu,snr,eta)

F = @(x) 1/sqrt(2*pi)*(e*exp(-(x-sqrt(eta*mu*snr)).^2/2)+(1-e)*exp(-x.^2/2)).*log2(1/sqrt(2*pi)*(e*exp(-(x-sqrt(eta*mu*snr)).^2/2)+(1-e)*exp(-x.^2/2)));
y = -quadgk(F,-20,20)-1/2*log2(2*pi*exp(1))+e*mu/2*(eta-1-log2(eta));   

return;