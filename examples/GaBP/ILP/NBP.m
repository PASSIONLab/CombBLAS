
% Non-parametric belief propagation code
% Written by Danny Bickson.
% updated: May-2009
%
%
% input: G - inverse covariance sqaure matrix size n x n containing edge weights. G_ij is inverse 
% covariance between nodes i and j. Note that unlike LDLC, where the
% decoding matrix H is sparse, we assume that the matrix G is sparse.
% self_pot - an array of kde() objects containing the self potentials. The
% self potentials are Gaussian mixtures
% max_iter - maximal number of iterations
% epsilon - convergence threshold
% output: x - a kde of size nx1 containing the beliefs
%

function [x,p]=NBP(G,self_pot,rounds,epsilon,type)
   
    density = 30; % max number of Gaussians in each mixture
    n = size(G,1); % number of check nodes
    assert(size(G,2) == size(G,1));
    
    % variable messages
    M = cell(n,n);

    %iterate
   for k=1:rounds

      %variable to check nodes
      % for each check
      for i=1:n
         % for each var
           for j=1:n
               %if there is an edge from var node i to check node j
               if ((G(i,j) ~= 0) && (i~=j))
                   
                  if (k == 1)
                      M{i,j} = integral_ekde(self_pot(i), G(i,j));
                  else      
                      % node degree
                      dd = sum(G(:,i)~= 0) - 1;
                      toMul = cell(1, dd);
                      cc = 1;
                      for l=1:n
                          if ((G(l,i) ~= 0) && (l~=j) && (l~= i))
                              current = M{l,i};
                              verify_kde(current);
                              toMul{cc} = current;
                              cc = cc + 1;
                          end
                      end % for l
                      toMul{cc} = self_pot(i);% 

                    % computes the approx. product of incoming mixtures
                     prod = kde_prod(density, toMul,type,epsilon);
                     M{i,j} = integral_ekde(prod, G(i,j));
                  end
                  
                  
               end % if
           end % for j
      end % for i

           disp(['NBP iteration ', num2str(k)]);      
   end
           
  %variable to check nodes
      % for each check
      x=cell(1,n);
      
      for i=1:n
         % for each var
               %if there is an edge from var node i to check node j
                
                 % node degree
                  dd = sum(G(:,i)~= 0) - 1;
                  toMul = cell(1, dd+1);
                  cc = 1;
                  for l=1:n
                      if ((G(l,i) ~= 0) && (l~=i))
                          current = M{l,i};
                          verify_kde(current);
                          toMul{cc} = current;
                          cc = cc + 1;
                      end
                  end % for l
                  toMul{cc} = self_pot(i);% 

                  % computes the approx. product of incoming mixtures
                  prod = kde_prod(density, toMul,type,epsilon);
                  x{i} = prod;
                 
      end % if
  
end

