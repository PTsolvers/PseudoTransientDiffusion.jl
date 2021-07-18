clear
%% symbolic functions and variables
syms H(t,x) q(t,x)
syms D Lx tau_r rho Re H0 k positive
syms lambda_k
%% governing equations
eq1      =   rho*diff(H(t,x),t)          +   diff(q(t,x),x);               % mass balace
eq2      = tau_r*diff(q(t,x),t) + q(t,x) + D*diff(H(t,x),x);               % momentum balance
%% equation for H
eq_H     = expand(diff(eq1,t)*tau_r - diff(eq2,x) + eq1);
%% scales and nondimensional variables
V_p      = sqrt(D/rho/tau_r);                                              % velocity scale - wave velocity
rho      = solve(Re == rho*V_p*Lx/D,rho);                                  % density from Reynolds number Re
%% dispersion relation
H(t,x)   = H0*exp(-lambda_k*V_p*t/Lx)*sin(pi*k*x/Lx);                      % Fourier term
disp_rel = expand(subs(subs(eq_H/H(t,x))));
cfs      = coeffs(disp_rel,lambda_k);
disp_rel = collect(simplify(disp_rel/cfs(end)),[lambda_k,pi,k]);
cfs      = coeffs(disp_rel,lambda_k);
%% optimal iteration parameters
a        = cfs(3);
b        = cfs(2);
c        = cfs(1);
discrim  = b^2 - 4*a*c;                                                    % quadratic polynomial discriminant
Re_opt   = solve(subs(discrim,k,1),Re,'PrincipalValue',true);
%% evaluate the solution numerically
num_cfs  = matlabFunction(fliplr(subs(cfs,k,1)));
Re1      = linspace(0,2*double(Re_opt),201);                               % create uniform grid of Re values
num_lam  = arrayfun(@(Re)(min(real(roots(num_cfs(Re))))),Re1);             % compute minimum of real part of roots
%% plot the spectral abscissa
figure(1);clf;colormap cool
plot(Re1,num_lam,'r-','LineWidth',1);axis square;grid on
xlim([min(Re1) max(Re1)])
xline(double(Re_opt),'k--','LineWidth',2,'Alpha',1)
ax = gca;
xlabel('$Re$','Interpreter','latex');ylabel('$\mathrm{max}\{\Re(\lambda_k)\}$','Interpreter','latex')
xticks([0 pi 2*pi 3*pi 4*pi])
xticklabels({'$0$','$\pi$','$2\pi$','$3\pi$','$4\pi$'})
ax.XAxis.TickLabelInterpreter = 'latex';
ax.YAxis.TickLabelInterpreter = 'latex';
set(gca,'FontSize',14)
title(['$' latex(disp_rel) '$'],'Interpreter','latex','FontSize',16)