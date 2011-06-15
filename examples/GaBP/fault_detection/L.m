% Code written by Harel Avissar and Danny Bickson
% Suplamentary material for the paper: 
% Fault identification via non-parametric belief propagation.
% By D. Bickson, D. Baron, A. Ihler, H. Avissar, D. Dolev
% In IEEE Tran of Signal Processing.
% http://arxiv.org/abs/0908.2005
% Code available from http://www.cs.cmu.edu/~bickson/gabp/

%function to compute a likelihood of a certain solution
function Likelihood=L(H, p, n, x, y, sigma)
    Likelihood = (1/(2*sigma^2))*trace(H'*H*x*x')+(log((1-p)/p)*ones(n,1)-(1/sigma^2)*H'*y)'*x;
end