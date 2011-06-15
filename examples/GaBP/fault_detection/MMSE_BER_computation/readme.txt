readme.m
% Fault identification via non-parametric belief propagation.
% By D. Bickson, D. Baron, A. Ihler, H. Avissar, D. Dolev
% http://arxiv.org/abs/0908.2005
% In IEEE Tran on Signal PRocessing
% Code available from http://www.cs.cmu.edu/~bickson/gabp

This file details the different Matlab routines used to
compute the MMSE bound that appears in the above paper.

Files:
mmse.m - computes MMSE for scalar Bernoulli random variable measured 
 with a scalar measurement with additive Gaussian noise and some SNR
ber.m - similar to mmse.m but bit error rate
constraint.m - a constraint used to choose among several possible 
 solutions to the fixed point expression of Guo et al.
efficiency_db.m - function that computes the multi-user efficiency
 (or degradation) of the scalar channel caused by other users
main.m - computes BER and MMSE bounds for range of parameters
ifnotfound.quadgk.m - supports older Matlab platforms

Notation:
n - length of (unknown) vector input
m - length of noisy measurements vector
e or epsilon - signal is Bernoulli random variable (RV) with parameter 
 epsilon
mu - ratio of m/(epsilon*n) - number of measurements per active element
snr - signal to noise ratio (either of individual RV or vector input)
eta - degradation of scalar channel (see Guo et al.)
q - percentage of nonzeros in the matrix 
sigma - amplitude of Gaussian noise in vector measurmeent system

Related papers:

D. Guo, D. Baron, and S. Shamai (Shitz), ``A single-letter 
characterization of optimal noisy compressed sensing,'' Proc. 
Allerton Conf. Commun., Control, and Computing, Monticello, IL, USA, 
Oct. 2009.

C.-C. Wang and D. Guo, ``Belief propagation is asymptotically equivalent 
to MAP detection for sparse linear systems,'' Allerton Conference on 
Communication, Control, and Computing, Sep. 2006. 

D. Guo and S. Verdu, ``Spectral efficiency of large-system CDMA via 
statistical physics,'' Conference on Information Sciences and Systems, 
Baltimore, MD USA, March 2003

Specific files appear below:

mmse.m
Inputs:
 e - probability that Bernoulli random variable (RV) is nonzero
 snr - signal to noise ratio of additive scalar Gauassian noise
Output: y the minimum mean square error (MMSE) for estimating the RV
This function computes the MMSE for a Bernoulli RV with parameter e 
measured with snr. The function uses F as a sub-function to compute 
an integral; the function then computes formula (14) in the 2009 
paper by Guo, Baron, and Shamai. 
NOTE: this function was written by Dongning Guo.

ber.m
Inputs:
 e - parameter of Bernoulli random variable 
 snr - SNR of Gauassian noise
Output: y the bit error rate acievable in estimating the RV
This function computes the BER by computing the Gaussian error 
function for two tail events.

constraint.m
Inputs:
 e - parameter of Bernoulli RV
 mu - ratio of m/(epsilon*n)
 snr - signal to noise ratio of vector measurement system
 eta - degradation (see Guo et al.)
Output: expression on top of page 5 in 2009 paper by Guo, Baron,
and Shamai.
This expression is used to choose the best solution to the fixed 
point solution (sometimes there are multiple solutions).
NOTE: this function was written by Dongning Guo.

efficiency_db.m
Inputs:
 e - parameter of Bernoulli input vector
 mu - ratio of m/(epsilon*n)
 snr - SNR of vector measurement system
Output: computes the efficiency eta of the vector measurmeent system
The computation uses the fixed point formula by Guo et al.
NOTE: Dror Baron adapted a previou function written by Dongning Guo.

main.m - script file that computes BER and MMSE bounds for range
of epsilon, where m, n, q, and sigma are fixed. (See notation as
described above.)

ifnotfound.quadgk.m
function that uses quadl on computer matlab platforms where the
(relatively new) quadgk function isn't supported.
