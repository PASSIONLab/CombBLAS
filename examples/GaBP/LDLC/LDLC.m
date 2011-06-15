
% Extended LDLC decoder
% Written by Danny Bickson.
% updated: 18-Dec-2008
%
% Supplamentray material of the paper: 
% "Low density lattice decoder via non-parametric belief propagation"
% By D. Bickson, A. T. Ihler, and D. Dolev,
% http://arxiv.org/abs/0901.3197
%
% input: H - NON SQUARE sparse parity matrix size nv x nc (nv = number of
% variable nodes, nc = number of check nodes)
% y - observation vector
% rounds - maximal number of iterations
% epsilon - convergence threshold
% sigma - sigma^2, the assumed noise level

% output: x - vector of size nx1, which is the solution to linear systems
% of equations H x = y, s.t. x is integer
%
% Acknowledgement: Thanks to N. Sommer and E. N. Hoch for their help in implementing the
% LDLC decoder.

function [x]=LDLC(H,y,rounds,epsilon,sigma)
   
    options = [-1 1]; % the allowed integers
    density = 30; % max number of Gaussians in each mixture
    
    nc = size(H,2); % number of check nodes
    nv = size(H,1); % number of variable nodes
    
    % variable messages
    M = cell(nv,nc);
    % check node messages
    MT = cell(nv,nc);

    % array holding variance (for convergence detection)
    v = zeros(nv,nc);
    % variance from last round
    old_v = v;

    
    %iterate
   for k=1:rounds

      %variable to check nodes
      % for each check
      for i=1:nc
         % for each var
           for j=1:nv
               %if there is an edge from var node i to check node j
               if (H(j,i) ~= 0)
                  dummy = kde(rand(1,density),1);  
                  if (k == 1)
                      M{j,i} = kde(y(i), sqrt(sigma), 1);% 
                  else
                      % node degree
                      dd = sum(H(:,i)~= 0);
                      toMul = cell(1, dd);
                      cc = 1;
                      valp = 1/sigma;
                      for l=1:nv
                          if (H(l,i) ~= 0 && l~=j)
                              toMul{cc} = MT{l,i};
                              valp = valp + 1/vt(l,i);
                              cc = cc + 1;
                          end
                      end % for l
                      toMul{cc} = kde(y(i), sqrt(sigma), 1);

                      valp = 1/valp;
               
                      % computes the approx. product of incoming mixtures
                      %prod = productApprox(dummy, toMul, {}, {}, 'epsilon', epsilon);
                      prod = productApprox(dummy, toMul, {}, {}, 'import', 2);
%                       if (length(getPoints(prod)) > density)
%                         prod = resample(prod, density, 'discrete');
%                       end
                      prod = kde(getPoints(prod), ones(1, length(getPoints(prod))) * sqrt(valp));
                      M{j,i} = prod;
                      assert (sum(isnan(getPoints(M{j,i}))) == 0);
                      assert (length(unique(getPoints(M{j,i}))) ~= 0);
                  end
                  
                  if (length(unique(getBW(M{j,i}))) ~= 1)
                    assert(length(unique(getBW(M{j,i}))) == 1);
                  end
                      v(j,i) = unique(getBW(M{j,i}).^2);
               end % if
           end % for j
      end % for i

      disp(['LDLC iteration ', num2str(k)]);      

       %check nodes to variable

       vt = zeros(nv,nc);
       mt = zeros(nv,nc);
       %for each var node
       for i=1:nv
           %for each check node
           for j=1:nc
               if (H(i,j) ~= 0)
                  points = [];
                  bandwidth = 0;
                  
                  for l=1:nc
                      if (H(i,l) ~= 0 && l~=j)
                          current = M{i,l}
                          assert (sum(isnan(getPoints(current))) == 0);
                          assert (length(unique(getPoints(current))) ~= 0);

                          if (length(points) == 0)
                            points = zeros(1, length(getPoints(current)));
                          end

                          if (length(points) ~= length(getPoints(current))) 
                              assert (length(points) == length(getPoints(current)));
                          end
                          points = points + (H(i,l) * getPoints(current));
                          bandwidth = bandwidth + H(i,l)^2 * unique(getBW(current).^2);
                          assert (length(unique(getBW(current))) == 1);
                          assert (unique(getBW(current).^2) == v(i,l));
                      end
                  end % for l
                  
                  points = points ./ H(i,j);
                  bandwidth = bandwidth / (H(i,j)^2);

                  % periodic shift operation
                  aPoints = [];
                  for l=1:length(options)
                      aPoints = [aPoints (options(l)/H(i,j) - points)];
                  end % for l
                      
                  toAdd = kde(aPoints, sqrt(bandwidth) * ones(1,length(aPoints)));

                  MT{i,j} = toAdd; 
                  if (length(unique(getBW(MT{i,j}))) ~= 1)
                      assert(length(unique(getBW(MT{i,j}))) == 1);
                  end
                  vt(i,j) = unique(getBW(MT{i, j}).^2);
               end % if
           end
       end

    
       % Stage 3 - convergence detection
       sum(sum((v - old_v).^2))
         if (sum(sum((v - old_v).^2)) < epsilon)
             disp(['LDLC factor converged in round ', num2str(k)]);
             break;
         end
         
         old_v = v;
    end
    
    
      %final decision
      % for each check
      for i=1:nc
          % node degree
          dd=sum(H(:,i)~=0);
          if (dd == 0)
              continue;
          end
          toMul = cell(1, dd+1);

          cc = 1;
          % for each var
          for l=1:nv
              if (H(l,i) ~= 0)
                  assert( vt(l,i) ~= 0);
                  toMul{cc} = MT{l,i};
                  cc = cc + 1;
              end
          end % for l

          toMul{cc} = kde(y(i), sqrt(sigma), 1);

          dummy = kde(rand(1,density),1);  
          % final approx. product of all incoming messages
          %prod = productApprox(dummy, toMul, {}, {}, 'epsilon', epsilon);
          prod = productApprox(dummy, toMul, {}, {}, 'import', 2);
          assert (sum(isnan(getPoints(prod))) == 0);
          assert (min(unique(getPoints(prod)) ~= 0));
         
          ret =prod;
          % takes max value of the mixture as the solution
          mm(i) = max(ret);

      end % for i

      %disp('LDLC result before rounding');
      mm = H*mm';
      %disp('LDLC result after rounding');
      % change this line to round(mm) in case the tranmission is integers
      x = sign(mm);
      
     v;
end

