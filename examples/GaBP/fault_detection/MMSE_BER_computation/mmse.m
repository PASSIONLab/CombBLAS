% MMSE of estimating a signal through a Gaussian channel
% Input signal is Bernoulli(e)

function y = mmse(e,snr)

F = @(x) real(exp(-x.^2/2)./(e*exp(-x*sqrt(snr)+snr/2)+(1-e)*exp(-2*x*sqrt(snr)+snr)));
y = e-e^2/sqrt(2*pi)*quadgk(F,-15,max(sqrt(snr)+6,15));
if y<0 | y>1,
    y
    e
    snr
    error( 'mmse out of range' )
end
%y = e-e^2/sqrt(2*pi)*quadgk(F,-15,max(sqrt(snr)+6,15));
return;