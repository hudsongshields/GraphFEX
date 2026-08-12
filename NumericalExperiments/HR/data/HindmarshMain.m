% Copyright 2020, All Rights Reserved
% Code by Tingting Gao and  Gang Yan*
% For paper, "Autonomous inference of complex network dynamics"
% by Tingting Gao, Gang Yan*

% This code runs simulations of Hindmarsh Rose nueronal dynamics, and construct
% a comprehensive library matrix for self- and interaction dynamics. The simulated time-series
% data and the library matrix are saved as two csv files. 


clc, clf, clear, close all
degree = 7.1;
% rng(10);
rng(55); % rng for second run
tic
disp('   HR-preprocess starts')

%% Parameters setting and import initial positions file and adjacency matrix file.
gc=0.15; %coupling strength
%import adjacency matrix

% filePath = .\BA_Nnodes100_Adj_deg_7_1.csv
degree_str = strrep(sprintf('%.1f', degree), '.', '_');
filePath = sprintf('./BA_Nnodes100_Adj_deg_%s.csv', degree_str);
% Read the CSV file
A = csvread(filePath);
%counting edges
b1 = (A~=0);
Num = sum(b1(:));
%number of nodes
n1 = length(A);

dim = 3;
n=dim*n1;
%% Simulation
x0= rand(1,n);
%x0 = csvread('initialPosition_HR.csv');
tspan=(0.01:0.01:100);
T = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[~,x]=ode45(@(t,x) hindmarshrose(x,A,n1,gc),tspan,x0);
degree_str = strrep(sprintf('%.1f', degree), '.', '_');
% filePath = sprintf('../new_data/HR_timeseries_BA_deg_%s.csv', degree_str);
filePath = sprintf('../new_data/HR_timeseries_BA.csv');
% Save the time series data to the file
csvwrite(filePath, x);

%%

SNR_values = [45, 40];  

degree_str = strrep(sprintf('%.1f', degree), '.', '_');
% filePath = sprintf('../new_data/HR_timeseries_BA_deg_%s.csv', degree_str);
filePath = sprintf('../new_data/HR_timeseries_BA.csv');
x = csvread(filePath);

% Function to add noise to data based on SNR
add_noise_to_data = @(data, SNR) ...
    data + sqrt(var(data(:)) / (10^(SNR / 10))) * randn(size(data));


for SNR = SNR_values
    signal_power = mean(var(x, 0, 1));
    noise_power = signal_power / (10^(SNR / 10));
    noise_std = sqrt(noise_power);
    noise = noise_std * randn(size(x));
    x_noisy = x + noise;

    SNR_str = sprintf('%d', SNR);
    noisy_filePath = sprintf('../new_data/HR_timeseries_BA_deg_%s_SNR_%s.csv', degree_str, SNR_str);
    csvwrite(noisy_filePath, x_noisy);


    disp(['Writing noisy file: ', noisy_filePath]);
end

%% derivative
delt = tspan(2)-tspan(1);
dy = zeros(size(x(3:T-2,:),1),n);
j = 1;
for t = 3:T-2
    for i = 1:n1
    dy(j,3*i-2)=(8*(x(t+1,3*i-2)-x(t-1,3*i-2))+x(t-2,3*i-2)-x(t+2,3*i-2))/(12*delt);
    dy(j,3*i-1)=(8*(x(t+1,3*i-1)-x(t-1,3*i-1))+x(t-2,3*i-1)-x(t+2,3*i-1))/(12*delt);
    dy(j,3*i)=(8*(x(t+1,3*i)-x(t-1,3*i))+x(t-2,3*i)-x(t+2,3*i))/(12*delt);
    end
    j = j+1;
end
xnew = x(3:T-2,:);
%% Build library and map data to construct elementray functions matrix
abs_A = abs(A);
polynomial = 3;
usesin = 1;
seconddiffX = 0;
seconddiffY = 0;
seconddiffZ = 0;
dim = 1;
[yout,ystring] = LibraryThree(xnew,dy,abs_A,polynomial,usesin,seconddiffX,seconddiffY,seconddiffZ,dim,0,size(xnew,1));
columns = ystring;
Tt = size(xnew,1);
kin = sum(A,2);
k_zero = 0;
for i = 1:length(kin)
    if kin(i)==0
        yout((i-1)*Tt+1:i*Tt,:) = zeros(Tt,size(yout,2));
        k_zero = k_zero+1;
    end
end
data = table(yout(:,1),yout(:,2),yout(:,3),yout(:,4),yout(:,5),...
    yout(:,6),yout(:,7),yout(:,8),yout(:,9),yout(:,10),...
    yout(:,11),yout(:,12),yout(:,13),yout(:,14),yout(:,15),...
    yout(:,16),yout(:,17),yout(:,18),yout(:,19),yout(:,20),...
    yout(:,21),yout(:,22),yout(:,23),yout(:,24),yout(:,25),...
    yout(:,26),yout(:,27),yout(:,28),yout(:,29),yout(:,30),...
    yout(:,31),yout(:,32),yout(:,33),yout(:,34),yout(:,35),...
    yout(:,36),yout(:,37),yout(:,38),yout(:,39),yout(:,40),...
    yout(:,41),yout(:,42),yout(:,43),yout(:,44),yout(:,45),...
    yout(:,46),yout(:,47),yout(:,48),yout(:,49),yout(:,50),...
    yout(:,51),yout(:,52),yout(:,53),yout(:,54),yout(:,55),...
    yout(:,56),yout(:,57),yout(:,58),'VariableNames', columns);

col = {'dx','dy','dz'};
dxdt = zeros(n1*Tt,1);
dydt = zeros(n1*Tt,1);
dzdt = zeros(n1*Tt,1);
for i = 1:n1
   dxdt((i-1)*Tt+1:i*Tt,1) = dy(:,3*i-2);
   dydt((i-1)*Tt+1:i*Tt,1) = dy(:,3*i-1);
   dzdt((i-1)*Tt+1:i*Tt,1) = dy(:,3*i);
end

k_zero = 0;
for i = 1:length(kin)
    if kin(i)==0
        dxdt((i-1)*Tt+1:i*Tt,:) = zeros(Tt,size(dxdt,2));
        dydt((i-1)*Tt+1:i*Tt,:) = zeros(Tt,size(dydt,2));
        dzdt((i-1)*Tt+1:i*Tt,:) = zeros(Tt,size(dzdt,2));
        k_zero = k_zero+1;
    end
end

%data_dx = table(dxdt,dydt,dzdt,'VariableNames', col);
%writetable(data,'../results/HR_ElementaryFunctions_Matrix.csv');
%writetable(data_dx,'../results/HR_dX.csv');

toc

disp('   HR-preprocess finished!')