%-----------------------------------------------
% Original code by Shriram Sarvotham, 2006. 
% Cleaned up a bit by Dror Baron, November 2008.
% Full original code is found on: http://www.ece.rice.edu/~drorb/CSBP/
%-----------------------------------------------
% Code rewritten by Danny Bickson, March 2009.
% Impvoed accuracy of computation, added support for non binary matrices 
% Added support for arbitrary self potentials
% Bugfixes and debugging notes by Harel Avissar, April 2009.
% Impvoed accuracy of computation, and better support for low sampling 
% rates
% New code is avaialble from
% http://www.cs.huji.ac.il/labs/danss/p2p/gabp/index.html
%------------------------------------------------
% Nonparametric belief propagation implementation
% Channel model is: y = Ax + w, where w is AWGN noise ~ N(0,sigma_Z)
% priors of x and y are Gaussian mixtures
% Input:
% A - m x n - real linear transformation matrix
% x - n x 1 - hidden vector
% y - m x 1 - observation vector
% sigma_Z - noise level
% max_iter - maximal number of iterations
% dispind - a vector indices to display
% epsilon - small value to add to fft to avoid division by zero
% pdf_prior - prior of x
% noise_prior - prior of y
% xticks - points to sample the Gaussian mixture


function [xrecon, mrecon, srecon]=NBP(A,x,y,sigma_Z,max_iter,...
    dispind,epsilon,pdf_prior,noise_prior,xticks)

n=length(x);% signal length      
m=length(y);% measurement length
M2N = cell(m,n);
N2M = cell(m,n);
model_order = length(xticks);
delta=xticks(2)-xticks(1);
pdf_prior=ifftshift(pdf_prior);
xticks=ifftshift(xticks);
noise_prior=ifftshift(noise_prior);

verify_pdf(pdf_prior);
verify_pdf(noise_prior);

assert(max(abs(y)) < max(abs(xticks))); % observation should be inside the xticks bounds
% or else the pdf will be shifted outside. Need to increase boundx if this
% assertion fails. 

%---------------
% BP ITERATIONS 
%---------------
try 
    
for it=1:max_iter	

   %---------------
   % FORWARD ITERATION - from signal to measurement
   % Calculate the product of all incoming mixtures except the one the
   % message is sent to.
   % The product is simply point by point product
   %---------------
   for in=1:n
       
       % For each neighbor of i
      neighbors = find(A(:,in)~=0)';
      ln=length(neighbors); 
      assert(ln > 1);
      
      if (it==1) % initial round - send the signal prior
         for j=1:ln
            N2M{neighbors(j),in} = pdf_prior(:)';
         end
      else  % round >= 2        
         for j=1:ln 
            m_ji = M2N{neighbors(j),in} + epsilon;
            verify_pdf(m_ji);
            if (j == 1) % first time
               pdf_res=m_ji;
            else 
               pdf_res=mulpdf(pdf_res,m_ji);
               verify_pdf(pdf_res);
            end
         end
         pdf_res=mulpdf(pdf_res, pdf_prior);
         verify_pdf(pdf_res);
         [mrecon(in), srecon(in),xrecon(in)]=meanvarmaxpdf(pdf_res, xticks); % computes statistics
         for j=1:ln   % to send next message
            m_ji = M2N{neighbors(j),in}+epsilon;
            verify_pdf(m_ji);
           
            N2M{neighbors(j),in}= divpdf(pdf_res,m_ji,epsilon);
            verify_pdf(N2M{neighbors(j),in});
            %%%assert(sum(isnan(N2M{neighbors(j),in}))==0);
         end
      end
   end
   if (it >=2) % display and break on last iteration
      dispvec_anderrors_gabp(mrecon, A, y, dispind, x);
      if (it==max_iter)
         break;
      end
   end
   %---------------
   % BACKWARD ITERATION - from measurement to signal
   % Calculate convolution - which is a product in the FFT domain
   %---------------
   for in=1:m
      neighbors=find(A(in,:)~=0);
      ln=length(neighbors);
      assert(ln > 1);
      
      for j=1:ln % process neighbors
         m_ij = N2M{in,neighbors(j)};
         m_ij = pdf_integral(m_ij, model_order, A(in,neighbors(j)),xticks,delta,epsilon);
         %%%assert(sum(isnan(m_ij))==0);
         if (j == 1) % first time
            pdf_res_all = fft((m_ij));
            %%%assert(sum(isnan(pdf_res_all))==0);     
         else    
            pdf_res_all=mulpdf_fft(pdf_res_all, m_ij);
            %%%assert(sum(isnan(pdf_res_all))==0);
         end
      end
      
      %convolve with the self potential of the noise
      if (sigma_Z>epsilon)
         pdf_res_all=mulpdf_fft(pdf_res_all, noise_prior);
      end
      %%%assert(sum(isnan(pdf_res_all))==0);
      
      for j=1:ln   %To send next message
         m_ij = N2M{in,neighbors(j)};
         m_ij = pdf_integral(m_ij, model_order, A(in,neighbors(j)),xticks,delta,epsilon);
         verify_pdf(m_ij);
         %unconvolve with the message node from j to i
         pdf_res=divpdf_fft(pdf_res_all, m_ij, epsilon);
         %%%assert(sum(isnan(pdf_res))==0);
         % get back from the FFT to the real domain
         pdf_res=ifft((pdf_res));
         %%%assert(sum(isnan(pdf_res))==0);
         pdf_res=abs((pdf_res));
         verify_pdf(pdf_res);
         %%%assert(sum(isnan(pdf_res))==0);
         % compute y(i) - current mixture
         %pdf_res=shiftpdf_fft(pdf_res, y(i), delta, model_order);       
         pdf_res=shiftpdf(pdf_res, y(in), delta, xticks);  
         %%%assert(sum(isnan(pdf_res))==0);
         M2N{in,neighbors(j)} = pdf_integral(pdf_res, model_order, A(in,neighbors(j)),xticks,delta,epsilon);
         verify_pdf(M2N{in,neighbors(j)});
     end
   end
   
 
   
end


catch ME
   disp(['assert in iteration ', num2str(it)]);
   [j in it]
  %pdf_res
  rethrow(ME)
   return;
end
% Handle computation of integral.
% For edges with weight 1 - does not do anything
% For edges with weight -1 - reverses the mixture
% For real non zero edges, calculates interpolation
function [npdf]= pdf_integral(pdf, model_order, edge,xticks,delta,epsilon)
   assert(edge ~= 0);
    if (edge == -1)
       npdf=reverse_gabp(pdf,model_order);
   elseif  (edge == 1)
       npdf = pdf;
    else
       %old_mean = fft_max(xticks,pdf);
       old_mean = sum(xticks.*pdf);
       npdf= interp1(xticks, pdf, xticks./abs(edge));
       npdf(isnan(npdf)) = epsilon; 
       if (edge < 0)
           npdf = reverse_gabp(npdf, model_order);
       end
           
       npdf=npdf./sum(npdf);
       new_mean = sum(xticks.*npdf);
       if (length(new_mean) == 1 && (abs(new_mean - old_mean * edge) >= 2*delta))
           %assert(abs(new_mean - old_mean * edge) < delta);%DB TODO 
       end
   end
end
% point by point multiplication
function c=mulpdf(a,b)
c=a.*b;
if (sum(abs(c)) <= 0)
    disp('bug');
end
c=c./sum(abs(c));
end

% point by point division
function c=divpdf(a,b,epsilon)
% carefully prevent division by zero in the FFT domain
% values in the FFT domain MAY be negative
% in this case, take either epsilon or -epsilon as appropriate

%DB - code by harel which crushes
% tmp = sign(real(b));
% tmp2 = imag(b);
% tmp3=max(epsilon,abs(real(b))).*tmp+tmp2*1i;
% c=a./tmp3;
% c=c/max(sum(abs(c)),epsilon);
c=a./(b+epsilon);
c(isnan(c)) = epsilon;

end
% point by point multiplication in the FFT domain
 function c=mulpdf_fft(a,b)
tmp=fft((b));
c=mulpdf(a,tmp);
 end
 % point by point division in the FFT domain
function c=divpdf_fft(a,b,epsilon)
tmp=fft((b));
c=a./(tmp+epsilon);
%c(isnan(c)) = epsilon;
end
% sanity checks
function []=verify_pdf(pdf)
   assert(sum(pdf < 0)==0);
   assert(sum(isnan(pdf))==0);
   assert(sum(isinf(pdf))==0);
   assert(length(unique(pdf)) > 1);
   assert(sum(~isreal(pdf)) == 0);
end

function [m,sig,mp]=meanvarmaxpdf(pdf, xx)
    pdf=pdf/sum(pdf);
    [mm,ind]=max(pdf);
    mp=xx(ind);
    m=sum(xx.*pdf);
    sig=sqrt(sum(xx.*xx.*pdf)-m*m);
end

function [] =dispvec_anderrors_gabp (v, A, y, dispind, x)
    l=length(dispind);
    s='[';
    for i=1:l
      s=sprintf('%s %7.2f',s,v(dispind(i)));
    end
    s=sprintf('%s]',s);
    mv=A*v';
    er=norm(y-mv);
    tnorm = norm(x-v);
    cost = norm(A*v'-y)+sum(abs(v));
    s=sprintf('%s y-Ax=(%7.4f) x-v=(%7.4f) log(c)=(%7.4f)',s, er, tnorm,cost);
    disp(s);
end

% DB: Since the array is fftshifted, the first position is zero (odd sample size assumed), no need to
% reverse it. The other values are reversed, which means we computed
% integral with edge weight of -1.
function op=reverse_gabp(ip, m)
    op=[ip(1) ip(m:-1:2)];
end


%move the new mean to "Offset - old mean"
function pdf=shiftpdf(pdf, offset, delta, xx)

    %old_pos = fft_max(xx,pdf);
%     old_pos = sum(xx.*pdf);
    pdf = fliplr(pdf);
    verify_pdf(pdf);
    pdf = interp1(xx,pdf,xx-offset);
 
    pdf(isnan(pdf)) = 0;
    pdf = pdf./sum(pdf);
    %new_pos = fft_max(xx,pdf);
%     new_pos = sum(xx.*pdf);
%     if (length(old_pos) == 1 && (abs((offset - old_pos) - new_pos) >= 6*delta))
%            %assert(abs((offset - old_pos) - new_pos) < delta); %TODO DANNY
%     end
    %pdf = fftshift(pdf);
end

end