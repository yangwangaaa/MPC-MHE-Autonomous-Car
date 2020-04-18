% point stabilization + Multiple shooting
clear all
close all
clc

% CasADi v3.4.5
addpath('/home/akshit/Downloads/casadi-linux-matlabR2014b-v3.5.1')
import casadi.*

T = 0.2; %[s]
N = 150; % prediction horizon
lr = 1.7;
lf = 1.1;
rob_diam = 2;
v_max = 20; v_min = -v_max;

x = SX.sym('x'); y = SX.sym('y'); 
v = SX.sym('v'); psi = SX.sym('psi'); beta = SX.sym('beta');
states = [x;y;psi;v;beta]; n_states = length(states);

delta = SX.sym('delta'); a = SX.sym('a');% Control Input
controls = [a,delta]; n_controls = length(controls);
rhs = [v*cos(psi+delta);v*sin(psi+delta);(v*sin(beta))/lr; a; atan2(lr*tan(delta),lf + lr)]; % system r.h.s

f = Function('f',{states,controls},{rhs}); % nonlinear mapping function f(x,u)
U = SX.sym('U',n_controls,N); % Decision variables (controls)
P = SX.sym('P',n_states + n_states);
% parameters (which include the initial state and the reference state)

X = SX.sym('X',n_states,(N+1));
% A vector that represents the states over the optimization problem.

obj = 0; % Objective function
g = [];  % constraints vector

Q = zeros(5,5); Q(1,1) = 5;Q(2,2) = 5;Q(3,3) = 1; Q(4,4) = 1;Q(5,5) = 1; % weighing matrices (states)
R = zeros(2,2); R(1,1) = 0.5; R(2,2) = 0.05; % weighing matrices (controls)

st  = X(:,1); % initial state
g = [g;st-P(1:5)]; % initial condition constraints
for k = 1:N
    st = X(:,k);  con = U(:,k);
    obj = obj+(st-P(6:10))'*Q*(st-P(6:10));% + con'*R*con; % calculate obj
    st_next = X(:,k+1);
    f_value = f(st,con);
    st_next_euler = st+ (T*f_value);
    g = [g;st_next-st_next_euler]; % compute constraints
end
% Add constraints for collision avoidance
obs_x = 1.15; % meters
obs_y = 5; % meters
obs_diam = 2; % meters
for k = 1:N+1   % box constraints due to the map margins
    g = [g ; -sqrt((X(1,k)-obs_x)^2+(X(2,k)-obs_y)^2) + 1.2*(rob_diam/2 + obs_diam/2)];
end

obs_x2 = 4; % meters
obs_y2 = 7; % meters
obs_diam = 2; % meters
% for k = 1:N+1   % box constraints due to the map margins
%     g = [g ; -sqrt((X(1,k)-obs_x2)^2+(X(2,k)-obs_y2)^2) + (rob_diam/2 + obs_diam/2)];
% end
% make the decision variable one column  vector
OPT_variables = [reshape(X,5*(N+1),1);reshape(U,2*N,1)];

nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);

opts = struct;
opts.ipopt.max_iter = 2000;
opts.ipopt.print_level =0;%0,3
opts.print_time = 1;
opts.ipopt.acceptable_tol =1e-3;
opts.ipopt.acceptable_obj_change_tol = 1e-3;

solver = nlpsol('solver', 'ipopt', nlp_prob,opts);

args = struct;

args.lbg(1:5*(N+1)) = 0;  % -1e-20  % Equality constraints
args.ubg(1:5*(N+1)) = 0;  % 1e-20   % Equality constraints

args.lbg(5*(N+1)+1 : 5*(N+1)+ (N+1)) = -inf; % inequality constraints
args.ubg(5*(N+1)+1 : 5*(N+1)+ (N+1)) = 0; % inequality constraints



args.lbx(1:5:5*(N+1),1) = -5; %state x lower bound
args.ubx(1:5:5*(N+1),1) = 5; %state x upper bound
args.lbx(2:5:5*(N+1),1) = -50; %state y lower bound
args.ubx(2:5:5*(N+1),1) = 50; %state y upper bound
args.lbx(3:5:5*(N+1),1) = deg2rad(-55); %state psi lower bound
args.ubx(3:5:5*(N+1),1) = deg2rad(55); %state psi upper bound
args.lbx(4:5:5*(N+1),1) = v_min; %state v lower bound
args.ubx(4:5:5*(N+1),1) = v_max; %state v upper bound
args.lbx(5:5:5*(N+1),1) = deg2rad(-55);%-inf; %state beta lower bound
args.ubx(5:5:5*(N+1),1) = deg2rad(55);%inf; %state beta upper bound


args.lbx(5*(N+1)+1:2:5*(N+1)+2*N,1) = -1.39; %a lower bound
args.ubx(5*(N+1)+1:2:5*(N+1)+2*N,1) = 1.39; %a upper bound
args.lbx(5*(N+1)+2:2:5*(N+1)+2*N,1) = deg2rad(-55); %delta lower bound
args.ubx(5*(N+1)+2:2:5*(N+1)+2*N,1) = deg2rad(55); %delta upper bound
%----------------------------------------------
% ALL OF THE ABOVE IS JUST A PROBLEM SET UP


% THE SIMULATION LOOP SHOULD START FROM HERE
%-------------------------------------------
t0 = 0;
x0 = [2 ; 0 ; 0.0; 0.0; 0.0];    % initial condition.
xs = [2 ; 10 ; 0.0; 0.0;0.0]; % Reference posture.

xx(:,1) = x0; % xx contains the history of states
t(1) = t0;

u0 = zeros(N,2);        % two control inputs for each robot
X0 = repmat(x0,1,N+1)'; % initialization of the states decision variables

sim_tim = 50; % Maximum simulation time

% Start MPC
mpciter = 0;
xx1 = [];
u_cl=[];

% the main simulaton loop... it works as long as the error is greater
% than 10^-6 and the number of mpc steps is less than its maximum
% value.
while(norm((x0-xs),2) > 1e0 && mpciter < sim_tim / T)
    args.p   = [x0;xs]; % set the values of the parameters vector
    % initial value of the optimization variables
    args.x0  = [reshape(X0',5*(N+1),1);reshape(u0',2*N,1)];
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
        'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);
    u = reshape(full(sol.x(5*(N+1)+1:end))',2,N)'; % get controls only from the solution
    xx1(:,1:5,mpciter+1)= reshape(full(sol.x(1:5*(N+1)))',5,N+1)'; % get solution TRAJECTORY
    u_cl= [u_cl ; u(1,:)];
    t(mpciter+1) = t0;
    % Apply the control and shift the solution
    [t0, x0, u0] = shift(T, t0, x0, u,f);
    xx(:,mpciter+2) = x0;
    X0 = reshape(full(sol.x(1:5*(N+1)))',5,N+1)'; % get solution TRAJECTORY
%     % Shift trajectory to initialize the next step
    X0 = [X0(2:end,:);X0(end,:)];
    mpciter;
    mpciter = mpciter + 1;
end;
xx_t = xx';
ss_error = norm((x0-xs),2)
% average_mpc_time = main_loop_time/(mpciter+1);
save('Kinematic_data','xx(1:2)')
Draw_MPC_Custom_PS_Obstacles(t,xx,xx1,u_cl,xs,N,rob_diam,obs_x,obs_y,obs_diam,obs_x2, obs_y2)