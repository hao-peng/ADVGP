% param = [log(sn); log(ell); log(sf); B; m; cholS]
% predictive distribution
function [mu, s2] = pred(param, xs, M)
[NS,D] = size(xs);
sn2 = exp(2*param(1));
inv_ell = exp(-param(2:D+1));
%eta = inv_ell.^2;
sf2 = exp(2*param(D+2));
B = reshape(param(D+3:D+2+M*D), M, D);
m = param(D+3+M*D:M+D+2+M*D);
cholS = zeros(M);
upper_ind = triu(ones(M))==1;
cholS(upper_ind) = param(M+D+3+M*D:M+D+2+M*D+M*(M+1)/2);
S = cholS'*cholS;

fixed_jitter = 1e-6; % jitter

B_invEll = scale_cols(B, inv_ell);
xs_invEll = scale_cols(xs, inv_ell);
Kbb = sf2*exp(-0.5*sq_dist(B_invEll')) + fixed_jitter*eye(M);
Kbb = (Kbb+Kbb')/2;
Ksb = sf2*exp(-0.5*sq_dist(xs_invEll', B_invEll'));

cholInvKbb = chol(inv(Kbb));
Phis = Ksb*cholInvKbb';

mu = Phis * m;
s2 = sn2 + sf2 - sum(Phis.*Phis,2) + sum(Phis.*(Phis*S),2);
end