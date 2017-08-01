% This is a simple demo for ADVGP implemented in Matlab without the 
% delayed/distributed updates for illustration only.
% Our experiments in the paper used the code implemented in C++
% on the ps-lite platform.
% We will release our code uppon paper acceptance.
clear; close; clc;
load('data/syn.mat'); %data from Snelson and Ghahramani  (2006)
addpath('util');

rng(0);

[N, D] = size(x);
M = 6;
step_size = 1e-3;
nIter = 500;

lik = log(1);
cov = log([ones(D,1);1]);
[~, B] = fkmeans(x', M); % using kmeans to initial inducing inputs

m = zeros(M,1);
cholS = eye(M);

upper_ind = triu(ones(M))==1;
param = [lik;cov;reshape(B,D*M,1);m;reshape(cholS(upper_ind),M*(M+1)/2,1)];

[f, df] = evidence(param, x, y, M);
[mu,s2] = pred(param,xtest,M);

figure(1);
scatter(x, y, 'r.');
hold on;
l1 = line(xtest, mu);
l2 = line(xtest, mu+2*sqrt(s2), 'Color', 'y');
l3 = line(xtest, mu-2*sqrt(s2), 'Color', 'y');
B = reshape(param((D+3):(D+2+M*D)), M, D);
l4 = line(B, -2*ones(M,1), 'Marker', '+', 'LineStyle', 'None');
xlim([min(x)-1, max(x)+1]);
ylim([min(y)-1, max(y)+1]);
hold off;

for iter = 1:nIter
  [g, dg] = workerUpdate(param, x, y, M);
  [f, param] = masterUpdate(param, g, dg, step_size, D, M);
  
  fprintf('iter %d: %f\n', iter, f);
  
  [mu,s2] = pred(param,xtest,M);
  set(l1,'YData',mu);
  set(l2,'YData',mu+2*sqrt(s2));
  set(l3,'YData',mu-2*sqrt(s2));
  B = reshape(param((D+3):(D+2+M*D)), M, D);
  set(l4,'XData',B);
  drawnow
end

