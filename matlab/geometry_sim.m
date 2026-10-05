% geometry_sim.m
% Top-view geometry of a hanging polar plotter
W = 700; H = 600;        % mm
ML = [0, H]; MR = [W, H];% motor positions
% sample path: a lissajous curve
t = linspace(0,2*pi,1500);
cx = W/2; cy = H/2;
ax = W*0.35; ay = H*0.30;
x = cx + ax*sin(3*t);
y = cy + ay*sin(4*t);

% compute string lengths
L1 = hypot(x-ML(1), y-ML(2));
L2 = hypot(x-MR(1), y-MR(2));

figure(1); clf; axis equal; hold on; box on;
plot([ML(1) MR(1)],[ML(2) MR(2)],'k-','LineWidth',3);
plot(x,y,'b-'); plot(cx,cy,'ro','MarkerFaceColor','r');
plot([ML(1) x(1)], [ML(2) y(1)], 'g--'); 
plot([MR(1) x(1)], [MR(2) y(1)], 'g--');
xlabel('X (mm)'); ylabel('Y (mm)'); title('Plotter Workspace & Sample Path');
legend('Top Bar','Path','Centre','Strings at start');
figure(2); clf;
plot(t,[L1;L2]'); xlabel('t'); ylabel('String length (mm)');
title('String Length Profiles'); legend('Left','Right');
