%
% % This Code can be obtained from: http://www.stanford.edu/~boyd/papers/fault_det.html
%
% Relaxed Maximum a Posteriori Fault Identification
% A. Zymnis, S. Boyd, and D. Gorinevsky
% Signal Processing, 89(6):989-999, June 2009. 
%
%
% Returned variables:
% X_amb: n by K matrix whose kth column is the 
%        kth element in the ambiguity group
% l_amb: K vector of losses in ambiguity group
% ERROR_FLAG: 0, if everything OK
%             1, if numerical errors occured
%


function [Xamb1 Lamb1 Xamb2 Lamb2 rounds]=Round_and_Local(s,K,H,p,n,y,sigma,LOFlag)
    %% Rounding
    [x_sort,ind_x] = sort(s,'descend');
    x_cand = []; l_cand = [];
    %x_sort
    for i = 1:min(max(K, 3*n*p),n)
        x_cur = -ones(n,1);
        x_cur(ind_x(1:i)) = 1;
        l_cur = L(H, p, n, x_cur, y, sigma);
        x_cand = [x_cand x_cur]; l_cand = [l_cand l_cur]; 
    end
    [l_sort,ind_l] = sort(l_cand,'ascend');
    X_amb = x_cand(:,ind_l(1:K)) ;%get ambiguity set
    L_amb = l_sort(1:K);
    Xamb1 = X_amb;
    Lamb1 = L_amb;
    
    iter = 0;
    if LOFlag
        %%LocalOpt
        x_cand = X_amb;
       l_cand = L_amb;
        EXIT_FLAG = 0;
        while(~EXIT_FLAG)
            x_cur = X_amb(:,1);x_best = x_cur;
            iter = iter+1;
            for j = 1:n
                x_cur(j) = -x_cur(j);
                if ismember(x_cur', x_cand', 'rows')
                    x_cur(j) = -x_cur(j);
                else
                    x_cand = [x_cand x_cur];
                    l_cand = [l_cand L(H, p, n, x_cur, y, sigma)];
                    x_cur(j) = -x_cur(j);
                end
            end
            [l_sort,ind_l] = sort(l_cand,'ascend');
            X_amb = x_cand(:,ind_l(1:K)); %get ambiguity set
            L_amb = l_sort(1:K);
            if all(x_best==X_amb(:,1))
                EXIT_FLAG = 1;
            end
        end
    end
    Xamb2 = X_amb;
    Lamb2 = L_amb;
    rounds = iter;    
end
