%Emeka Peters - 100953293
%ELEC 4700 PA10

is = 0.01 * (10^-12);
ib = 0.1 * (10^-12);
vb = 1.3;
gp = 0.1;

V = linspace(-1.95, 0.7, 200);

I = zeros(1, 200);
I2 = zeros(1, 200);

I = (is .* (exp(V .* 1.2/0.025) - 1)) + (gp .* V) - (ib .* (exp((-1.2/0.025) .* (V + vb))));
I2 = I + ((randn(1, 200))) .* 0.2 .* I; %((is .* (exp(1.2/0.025) - 1)) + (gp .* V) - ((ib .* (exp((-1.2/0.025) .* (V + vb))))));

figure(1);
plot(V, I);
title('I vs V');
figure(2);
plot(V, I2);
title('I2 vs V');

figure(3);
semilogy(V, I);
title('I vs V Log Scale');
figure(4);
semilogy(V, I2);
title('I2 vs V Log Scale');

% Question 2 
a = polyfit(I, V, 4);

b = polyfit(I, V, 8);
c = polyfit(I2, V, 4);
d = polyfit(I2, V, 8);

% Question 3 
fo = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3))');
ff = fit(V',I',fo);
If = ff(V);
figure(5);
plot(V, If);
hold on;

% Question 3A
fo = fittype('A.*(exp(1.2*x/25e-3)-1) + gp.*x - C*(exp(1.2*(-(x+vb))/25e-3))');
ff = fit(V',I',fo);
If = ff(V);
plot(V, If);
hold on;

% Question 3B
D = vb;
fo = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+vb))/25e-3))');

ff = fit(V',I',fo);
If = ff(V);
plot(V, If);
hold on;

% Question 3C
fo = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+vb))/25e-3)-1)');
ff = fit(V',I',fo);
If = ff(V);
plot(V, If);
hold on;

% Question 4 
inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net);
Inn = outputs;

% Question 5 
figure(6);
plot(V, Inn);

% Question 5 - B