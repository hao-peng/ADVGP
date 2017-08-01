% compute chol(inv(A)) for spd matrix A 
function U = chol_inv(A)
rot180   = @(A)   rot90(rot90(A));                     % little helper functions
%U = inv_triu(rot180(jitChol(rot180(A))'));
U = inv(rot180(chol(rot180(A))'));
end