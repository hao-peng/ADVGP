% proximal update for variational parameters
% gradient decent for other parameters
function [f, newParam] = masterUpdate(param, g, dg, step_size, D, M)
m = param(D+3+M*D:M+D+2+M*D);
cholS = zeros(M);
upper_ind = triu(ones(M))==1;
cholS(upper_ind) = param(M+D+3+M*D:M+D+2+M*D+M*(M+1)/2);
%S = cholS'*cholS;
%S_mm = S + m * m';

%h = 0.5*(-sum(log(diag(cholS).^2))-M+trace(S_mm));
h = 0.5*(-sum(log(diag(cholS).^2))-M+m'*m+sum(sum(cholS.*cholS)));
f = g + h;

newParam = zeros(size(param));
newParam(1:D+2+M*D) = param(1:D+2+M*D) - step_size *dg(1:D+2+M*D);

% normal update
%newParam(D+3+M*D:M+D+2+M*D) = param(D+3+M*D:M+D+2+M*D)-...
%  step_size*(dg(D+3+M*D:M+D+2+M*D)+m);
%cholS_dh_dS = cholS*(eye(M) - inv(S));
%newParam(M+D+3+M*D:M+D+2+M*D+M*(M+1)/2) = param(M+D+3+M*D:M+D+2+M*D+M*(M+1)/2)...
%  -step_size*(dg(M+D+3+M*D:M+D+2+M*D+M*(M+1)/2)...
%  +reshape(cholS_dh_dS(upper_ind),M*(M+1)/2,1));

% proximal update
newParam(D+3+M*D:M+D+2+M*D) = (param(D+3+M*D:M+D+2+M*D)-...
  step_size*dg(D+3+M*D:M+D+2+M*D))/(1+step_size);

newCholS = zeros(M);
newCholS(upper_ind) = param(M+D+3+M*D:M+D+2+M*D+M*(M+1)/2)-...
  step_size*dg(M+D+3+M*D:M+D+2+M*D+M*(M+1)/2);
diagNewCholS = diag(newCholS);
newCholS = newCholS/(1+step_size);
newCholS(eye(M)==1) = (diagNewCholS+sqrt(diagNewCholS.^2+4*(1+step_size)*step_size))/(2*(1+step_size));
newParam(M+D+3+M*D:M+D+2+M*D+M*(M+1)/2) = reshape(newCholS(upper_ind),M*(M+1)/2,1);
end