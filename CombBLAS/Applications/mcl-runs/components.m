function labels = components(Gin, D)
% CONNECTED COMPONENTS
%
% labels = components(G)
% labels = components(G, D)
%
% Input: G is an undirected (symmetric) graph
%      : D is a starting vector describing rooted trees if available.
%        (node-parent information). This argument is optional.
% Output: labels is a vector which contains the component number 
%         for each node.
%
% Viral Shah (C) 2005. All rights reserved.
%
% $Id: components.m 147 2007-04-25 02:42:42Z viral $

%if ~issym(G)
%  error ('Input graph not symmetric');
%end

G = zerodiag (spones (grsparse (Gin)));
n = length(G);
if nargin == 1
  D = (1:n)';
end

% The edges of G are {u, v}
[u v] = find (G);

while true 
  %  Perform conditional hooking
  D = conditional_hooking (D, u, v);

  %  Check for stars
  star = check_stars (D);

  % If all nodes are in a star, return, otherwise continue
  if nnz(star) == n;
    % Really check for termination - G_conn should be diagonal
    labels = fixuplabels (D);
    G_conn = zerodiag (contract (G, labels));
    if nnz(G_conn) == 0; return; end
  end 
  
  %  Perform unconditional hooking
  D = unconditional_hooking (D, star, u, v);

  % Perform pointer jumping
  D = pointer_jumping (D);
end

end % components()


%  Perform conditional hooking
function D = conditional_hooking (D, u, v)
  Du = D(u);
  Dv = D(v);

  hook = find (Du == D(Du) & Dv < Du);
  % Du == D(Du) star check might not be enough
  Du = Du(hook);
  Dv = Dv(hook);

  D(Du) = Dv;
end % conditional_hooking()


%  Perform conditional hooking
function D = unconditional_hooking (D, star, u, v)
  Du = D(u);
  Dv = D(v);
  
  hook = find (star(u) & Dv ~= Du);
  Du = Du(hook);
  Dv = Dv(hook);
  
  D(Du) = Dv;
end % unconditional_hooking()


%  Check for stars
function star = check_stars (D)
  n = length(D);
  star = ones (n, 1);

  notstars = find (D ~= D(D));
  star(notstars) = 0;
  Dnotstars = D(notstars);
  star(Dnotstars) = 0; %not needed
  star(D(Dnotstars)) = 0;

  star = star(D);
end % check_stars()


% Perform pointer jumping
function D = pointer_jumping (D)
  n = length(D);
  Dold = zeros(n,1);

  while any(Dold ~= D)
    Dold = D;
    D = D(D);
  end
end % pointer_jumping()
