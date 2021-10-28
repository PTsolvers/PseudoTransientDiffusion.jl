clear
%% symbolic functions and variables
syms H(tau,x) q(tau,x)
syms D Lx theta_r rho rho_ph dt Re Da H0 k positive
syms lambda_k
%% governing equations
eq1         =     rho*diff(H(tau,x),tau) + diff(q(tau,x),x) + rho_ph*H(tau,x)/dt; % mass balace
eq2         = theta_r*diff(q(tau,x),tau) + q(tau,x) + D*diff(H(tau,x),x);         % momentum balance
%% equation for H
eq_H        = expand(diff(eq1,tau)*theta_r - diff(eq2,x) + eq1);
%% scales and nondimensional variables
V_p         = sqrt(D/rho/theta_r);                                                % velocity scale - wave velocity
rho         = solve(Re == rho*V_p*Lx/D,rho);                                      % density from Reynolds number Re
dt          = solve(Da == Lx^2*rho_ph/D/dt,dt);
%% dispersion relation
H(tau,x)    = H0*exp(-lambda_k*V_p*tau/Lx)*sin(pi*k*x/Lx);                        % Fourier expansion term
disp_rel    = expand(subs(subs(eq_H/H(tau,x))));
cfs         = coeffs(disp_rel,lambda_k);
disp_rel    = collect(simplify(disp_rel/cfs(end)),[lambda_k,pi,k]);
cfs         = coeffs(disp_rel,lambda_k);
%% optimal iteration parameters
a           = cfs(3);
b           = cfs(2);
c           = cfs(1);
discrim     = b^2 - 4*a*c;                                                        % quadratic polynomial discriminant
Re_opt      = solve(subs(discrim,k,1),Re);Re_opt=Re_opt(1);
%% evaluate the solution numerically
fun_cfs     = matlabFunction(fliplr(subs(cfs,k,1)));
fun_Re_opt  = matlabFunction(Re_opt);
Da1         = linspace(1e-6,100,501);                                             % create 1D grid of Da values
Re1         = linspace(pi/2,5*pi,501);                                            % create 1D grid of Re values
[Re2,Da2]   = ndgrid(Re1,Da1);                                                    % create 2D grid of Re and Da values
num_lam     = arrayfun(@(Re,Da)(min(real(roots(fun_cfs(Da,Re))))),Re2,Da2);       % compute minimum of real part of roots
num_Re_opt  = fun_Re_opt(Da1);
num_lam_opt = arrayfun(@(Re,Da)(min(real(roots(fun_cfs(Da,Re))))),num_Re_opt,Da1);
%% plot the spectral abscissa
figure(1);clf;colormap cool
contourf(Re2,Da2,num_lam,10,'LineWidth',1);axis square;cb=colorbar;
hold on
plot(num_Re_opt,Da1,'w--','LineWidth',4)
hold off
ax = gca;
xlabel('$Re$','Interpreter','latex')
ylabel('$Da$','Interpreter','latex')
xticks([pi/2 pi 2*pi 3*pi 4*pi 5*pi])
xticklabels({'$\pi/2$','$\pi$','$2\pi$','$3\pi$','$4\pi$','$5\pi$'})
cb.Label.Interpreter          = 'latex';
cb.Label.String               = '$\mathrm{min}\{\Re(\lambda_k)\}$';
ax.XAxis.TickLabelInterpreter = 'latex';
ax.YAxis.TickLabelInterpreter = 'latex';
set(gca,'FontSize',14)
cb.Label.FontSize = 16;
title(['$' latex(disp_rel) '$'],'Interpreter','latex','FontSize',16)