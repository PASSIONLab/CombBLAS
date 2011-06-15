 function [prod] = kde_prod(density, toMul,type,epsilon)
        assert(nargin >= 3);
        
        dummy = kde(rand(1,density),1); 
         % computes the exact/approx. product of mixture array
          switch(type)
              case 'exact'              
                   prod = productExact(dummy,toMul,{},{});
              case 'epsilon'
                  prod = productApprox(dummy, toMul, {}, {}, 'epsilon', epsilon);
              case 'gibbs2'
                  prod = productApprox(dummy, toMul, {}, {}, 'gibbs2', epsilon);                  
              case 'import'
                  prod = productApprox(dummy, toMul, {}, {}, 'import', 2);
              otherwise
                  error('wrong type!');
          end
          
          verify_kde(prod);
    end