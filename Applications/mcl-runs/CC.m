function CC

i = [1 1 2 3 4 5]';
j = [3 5 3 4 5 4]';
v = [1 1 1 1 1 1]';
G = sparse(i,j,v);
G = G + G';

n = length(G);
D = (1:n)';
[u v] = find (G);

Du = D(u);
Dv = D(v);

hook=find (Du == D(Du) & Dv < Du);
Du = Du(hook);
Dv = Dv(hook);
D(Du) = Dv;

check_stars(D)
end

function star = check_stars (D)
  n = length(D);
  star = ones (n, 1);

  notstars = find (D ~= D(D));
  star(notstars) = 0;
  Dnotstars = D(notstars);
  star(Dnotstars) = 0;
  star(D(Dnotstars)) = 0;

  star = star(D);
end % check_stars()