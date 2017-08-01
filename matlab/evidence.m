% param = [log(sn); log(ell); log(sf); B; m; cholS]
% this my lowerbound used in ADVGP
% variational lower bound
function [f, df] = evidence(param, x, y, M)
[N,D] = size(x);
sn2 = exp(2*param(1));
inv_ell = exp(-param(2:D+1));
eta = inv_ell.^2;
sf2 = exp(2*param(D+2));
B = reshape(param(D+3:D+2+M*D), M, D);
m = param(D+3+M*D:M+D+2+M*D);
cholS = zeros(M);
upper_ind = triu(ones(M))==1;
cholS(upper_ind) = param(M+D+3+M*D:M+D+2+M*D+M*(M+1)/2);
S = cholS'*cholS;

fixed_jitter = 0;1e-6; % jitter

B_invEll = scale_cols(B, inv_ell);
x_invEll = scale_cols(x, inv_ell);
Kbb = sf2*exp(-0.5*sq_dist(B_invEll')) + fixed_jitter*eye(M);
Kbb = (Kbb+Kbb')/2;
Kxb = sf2*exp(-0.5*sq_dist(x_invEll', B_invEll'));

cholInvKbb = chol_inv(Kbb); %cholInvKbb = chol(inv(Kbb));
Phin = Kxb*cholInvKbb';

PhinPhin = Phin'*Phin;
Q = PhinPhin/sn2 + eye(M);
S_mm = S + m * m';

Psi = zeros(M);
Psi(upper_ind) = 1;
Psi(eye(M)==1) = 0.5;

B2 = B.*B;
x2 = x.*x;
E = -m*y'+(m*m'+S)*Phin'-Phin';
cholInvKbbE = cholInvKbb'*E;
cholInvKbbEDotKbx = cholInvKbbE.*Kxb';
F = (cholInvKbb'*((cholInvKbb*Kxb'*E').*Psi')*cholInvKbb).*Kbb;
FF = F + F';

% evidence
g = 0.5*N*log(2*pi)+0.5*N*log(sn2)+0.5/sn2*(y'*y-2*y'*Phin*m ...
  + sum(sum(PhinPhin.*S_mm)) + sf2*N - sum(sum(Phin.*Phin)));
h = 0.5*(-sum(log(diag(cholS).^2))-M+trace(S_mm));
f = g + h;

% gradient
df = zeros(size(param));

% df_dlnsn
df(1) = N-1/sn2*(y'*y-2*y'*Phin*m+sum(sum(PhinPhin.*S_mm))+...
  sf2*N-sum(sum(Phin.*Phin)));

% df_dlnell
% df(2:D+1) = -eta.*(2*ones(1,M)*(B.*(cholInvKbbEDotKbx*x))-...
%   ones(1,M)*cholInvKbbEDotKbx*x2-ones(1,N)*cholInvKbbEDotKbx'*B2-...
%   2*ones(1,M)*(B.*(F*B))+ones(1,M)*FF*B2)'/sn2;
df(2:D+1) = -eta.*(2*ones(1,M)*(B.*(cholInvKbbEDotKbx*x))-...
  ones(1,M)*cholInvKbbEDotKbx*x2-ones(1,N)*cholInvKbbEDotKbx'*B2-...
  ones(1,M)*(B.*(FF*B))+ones(1,M)*FF*B2)'/sn2;

% df_dlnsf
df(D+2) = 1/sn2*(-y'*Phin*m+sum(sum(PhinPhin.*S_mm))+...
  sf2*N-sum(sum(Phin.*Phin)));

% df_dB
df(D+3:D+2+M*D) = reshape((cholInvKbbEDotKbx*scale_cols(x,eta)-...
  (cholInvKbbEDotKbx*ones(N,1)*ones(1,D)).*scale_cols(B,eta)-...
  FF*scale_cols(B,eta)+(FF*ones(M,1)*eta').*B)/sn2, M*D, 1);

% df_dm
df(D+3+M*D:M+D+2+M*D) = -y'*Phin/sn2+m'*Q;

% df_dcholS
cholS_df_dS = cholS*(Q - inv(S));
df(M+D+3+M*D:M+D+2+M*D+M*(M+1)/2) = reshape(cholS_df_dS(upper_ind),M*(M+1)/2,1);
end