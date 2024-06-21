%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% How To Create a Smooth Velocity Profile (for takeoff)
%
% See https://jwdinius.github.io/blog/2018/eta3traj
%
% Parker Lusk
% 5 Sept 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear, clc;

%% Path description
ps = 0;
pe = 1;

%% Parameters
v_max = 0.35; v_min = -v_max;
a_max = 0.5; a_min = -a_max;
j_max = 1; j_min = -j_max;
% n.b., j_max >= a_max

% path length
stot = norm(ps - pe);

%% Initial Conditions
t0 = 0;
v0 = 0;
a0 = 0;
% n.b., v0 <= v_max; a0 <= a_max

% there is some constraint on a_max/j_max and v0/a0.
% also, depending on the initial conditions, some sections may not be
% necessary (e.g., it might be necessary to jump straight to Section 2 / 3)

%% Determine if Cruise Segment Exists
% Consider v_max as a free variable. Under time-optimal control, jerk would
% be bang-off-bang-off-bang and there would be no cruise period (since
% there is no velocity limit). The goal of the following code is to
% calculate what the maximum attainable velocity would be in this case of
% no velocity limit, but under accel and jerk limits. If vmax < v_max, then
% v_max must be capped to vmax and there will be no cruise period.
% Otherwise, there will be a cruise period, where velocity is limited to
% v_max.

% Section 1
dt1 = (a_max - a0) / j_max;
ds1 = v0*dt1 + 1/2*a0*dt1^2 + 1/6*j_max*dt1^3;
v1 = v0 + a0*dt1 + 1/2*j_max*dt1^2;
% Section 7
dt7 = -a_min/j_max;
ds7 = 1/6*j_max*dt7^3; % (by plugging in for v6)

% my quadratic eq
a1 = 1/a_max;
b1 = a_max/j_max;
c1 = ds1 + ds7 - stot - (5*a_max^3)/(24*j_max^2) - v1^2/(2*a_max);
vmax1 = max(roots([a1 b1 c1]));

% Joe's quadratic eq
a2 = 1/a_max;
b2 = (1*a_max)/(2*j_max);
% c2 = ds1 + ds7 - stot - (7*a_max^3)/(3*j_max^2) - (a_max*v1)/a_max - v1^2/a_max + a_max^3/(2*j_max^2) + v1/j_max + v1^2/(2*a_max^3);
c2 = ds1 + ds7 - stot - (7*a_max^3)/(3*j_max^2) - v1*(a_max/j_max + v1/a_max) + 1/(2*a_max)*(a_max^2/j_max+v1/a_max)^2;
vmax2 = max(roots([a2 b2 c2]));

if v_max > vmax1, v_max = vmax1; end
% if v_max > vmax2, v_max = vmax2; end

%% Compute Segment Timings of Velocity Profile

% Section 1: maximum jerk until max acceleration
dt1 = (a_max - a0) / j_max;
ds1 = v0*dt1 + 1/2*a0*dt1^2 + 1/6*j_max*dt1^3;
v1 = v0 + a0*dt1 + 1/2*j_max*dt1^2;

% Section 3: prelude
dt3 = -a_max/j_min;
dv3 = -1/2*j_min*dt3^2;

% Section 2: zero jerk, max accel, until *almost* max vel
v2 = v_max - dv3;
dt2 = (v2 - v1) / a_max;
ds2 = v1*dt2 + 1/2*a_max*dt2^2;

% Section 3: min jerk until zero accel and until max vel
ds3 = v2*dt3 + 1/2*a_max*dt3^2 + 1/6*j_min*dt3^3;

% Section 4: cruise; save for last

% Section 5: min jerk until min acceleration
dt5 = a_min/j_min;
ds5 = v_max*dt5 + 1/6*j_min*dt5^3;
v5 = v_max + 1/2*j_min*dt5^2;

% Section 7: prelude
dt7 = -a_min/j_max;
dv7 = -1/2*j_max*dt7^2;

% Section 6: min accel until *almost* zero vel
v6 = -dv7;
dt6 = (v6 - v5)/a_min;
ds6 = v5*dt6 + 1/2*a_min*dt6^2;

% Section 7: max jerk until zero accel and zero vel
ds7 = v6*dt7 + 1/2*a_min*dt7^2 + 1/6*j_max*dt7^3;

% Section 4: finale
s_sum = (ds1+ds2+ds3+ds5+ds6+ds7);
if s_sum < stot
    ds4 = stot - s_sum;
    dt4 = ds4/v_max;
else
    ds4 = 0;
    dt4 = 0;
end

%% Evaluate Kinematic Path

T = t0 + cumsum([dt1 dt2 dt3 dt4 dt5 dt6 dt7]);
s = ps + cumsum([ds1 ds2 ds3 ds4 ds5 ds6 ds7]);

N = 10000;
t = linspace(0, T(end), N);
p = zeros(1,N); v = zeros(1,N); a = zeros(1,N); j = zeros(1,N);

% Section 1
ind = t<T(1);
delta_t = t(ind) - t0;
p(ind) = ps + v0.*delta_t + 1/2.*a0.*delta_t.^2 + 1/6.*j_max.*delta_t.^3;
v(ind) = v0 + a0.*delta_t + 1/2.*j_max.*delta_t.^2;
a(ind) = a0 + j_max.*delta_t;
j(ind) = j_max;

% Section 2
ind = t>=T(1) & t<T(2);
delta_t = t(ind) - T(1);
p(ind) = s(1) + v1.*delta_t + 1/2*a_max.*delta_t.^2;
v(ind) = v1 + a_max.*delta_t;
a(ind) = a_max;
j(ind) = 0;

% Section 3
ind = t>=T(2) & t<T(3);
delta_t = t(ind) - T(2);
p(ind) = s(2) + v2.*delta_t + 1/2*a_max.*delta_t.^2 + 1/6.*j_min.*delta_t.^3;
v(ind) = v2 + a_max.*delta_t + 1/2*j_min.*delta_t.^2;
a(ind) = a_max + j_min.*delta_t;
j(ind) = j_min;

% Section 4
ind = t>=T(3) & t<T(4);
delta_t = t(ind) - T(3);
p(ind) = s(3) + v_max.*delta_t;
v(ind) = v_max;
a(ind) = 0;
j(ind) = 0;

% Section 5
ind = t>=T(4) & t<T(5);
delta_t = t(ind) - T(4);
p(ind) = s(4) + v_max.*delta_t + 1/6.*j_min.*delta_t.^3;
v(ind) = v_max + 1/2.*j_min.*delta_t.^2;
a(ind) = j_min.*delta_t;
j(ind) = j_min;

% Section 6
ind = t>=T(5) & t<T(6);
delta_t = t(ind) - T(5);
p(ind) = s(5) + v5.*delta_t + 1/2.*a_min.*delta_t.^2;
v(ind) = v5 + a_min.*delta_t;
a(ind) = a_min;
j(ind) = 0;

% Section 7
ind = t>=T(6) & t<=T(7);
delta_t = t(ind) - T(6);
p(ind) = s(6) + v6.*delta_t + 1/2.*a_min.*delta_t.^2 + 1/6.*j_max.*delta_t.^3;
v(ind) = v6 + a_min.*delta_t + 1/2.*j_max.*delta_t.^2;
a(ind) = a_min + j_max.*delta_t;
j(ind) = j_max;

%% Plotting

dp = diff([ps p]);
dv = diff([0 v]);
da = diff([0 a]);



figure(1), clf;
subplot(411); grid on; hold on;
plot(t,p); ylabel('position'); title('Motion Profile');
for i=1:length(T), plot([T(i) T(i)], [0 1], '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1); end
subplot(412); grid on; hold on;
plot(t,v); ylabel('velocity');
for i=1:length(T), plot([T(i) T(i)], [0 0.4], '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1); end
% plot(t,dp/mean(diff(t)));
subplot(413); grid on; hold on;
plot(t,a); ylabel('acceleration');
for i=1:length(T), plot([T(i) T(i)], [-0.5 0.5], '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1); end
% plot(t,dv/mean(diff(t)));
% plot(t,diff([0 dp])/mean(diff(t)).^2);
subplot(414); grid on; hold on;
plot(t,j); ylabel('jerk');
for i=1:length(T), plot([T(i) T(i)], [-1 1], '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1); end
% plot(t,da/mean(diff(t)));
% plot(t,diff([0 dv])/mean(diff(t)).^2);
% plot(t,diff([0 diff([0 dp])])/mean(diff(t)).^3);
xlabel('time [s]');