% send_spiral.m - MATLAB to Arduino serial demo (blocking)
port = serialport("auto",115200); % replace with actual port if needed
pause(2);

% simple spiral in mm
W=700; H=600; cx=W/2; cy=H/2;
rmax = min(W,H)/2*0.8;
theta = linspace(0,8*pi,1200);
r = linspace(10,rmax,1200);
x = cx + r.*cos(theta);
y = cy + r.*sin(theta);

% pack and send as lines: X<mm> Y<mm> P<0/1>\n
pen = zeros(size(x)); pen(1)=0; pen(20:end)=1; % pen down after some moves
for i=1:numel(x)
    cmd = sprintf("X%.1f Y%.1f P%d\n",x(i),y(i),pen(i));
    write(port, cmd, "string");
    pause(0.001); % pacing
end
write(port,"DONE\n","string");
