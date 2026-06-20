function [U, V] = RGD_NMF(X, Options)

[M, N] = size(X);
Data_num = length(Options.labels);  
n = Options.gndSmpNum;    
labels = Options.labels;
labels(n+1:end) = 0;     

W = Options.W;
if ~isempty(W)
    D = diag(sum(W, 2));
    L = D - W;  
end
       
label_matrix = zeros(Data_num, Options.KClass);  
for i =1:Data_num
    if labels(i)>0
        label_matrix(i,labels(i)) = 1;
    end
end

Q =label_matrix;
Q1=abs(Q-1);        
Q1(n+1:end,:) = 0;   
Q=Q';
Q1= Q1';
U = rand(M, Options.KClass);   
V = rand(N, Options.KClass);

fprintf('X dimensions: %d x %d\n', size(X));
fprintf('U dimensions: %d x %d\n', size(U));
fprintf('V dimensions: %d x %d\n', size(V));
fprintf('W dimensions: %d x %d\n', size(W));
fprintf('D dimensions: %d x %d\n', size(D));


for iters = 1:Options.maxIter
    residual = X - U * V';
    
    row_norms = sqrt(sum(residual.^2, 2));  
    row_norms(row_norms < eps) = eps;      
    I = diag(1 ./ row_norms);               
    
    numerator_U = I * X * V;
    denominator_U = I * U * (V' * V);
    U = U .* (numerator_U ./ (denominator_U + eps));
     
    numerator_V = X' * I * U + Options.beta * W * V;
    denominator_V = V * (U' * I * U) + Options.alpha * (Q1') + Options.beta * D * V;
    V = V .* (numerator_V ./ (denominator_V + eps));
end
end
