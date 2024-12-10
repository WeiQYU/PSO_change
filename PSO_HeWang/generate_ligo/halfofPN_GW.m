%% 初始化数据
G = 1; % 万有引力常数
c = 1; % 光速

Msun =1 ; % 太阳质量1.989e30,但是太阳质量本身就是一个单位
Mpc = 3.086e22; % Mpc转换为m

m1 = 10 * Msun; % 第一个天体质量
m2 = 25 * Msun; % 第二个天体质量
mu = (m1 * m2) / (m1 + m2); % 两个天体质量的组合
M = m1 + m2; % 天体总质量
eta = mu / M; % Symmetric mass ratio
DLS = 250 * Mpc;
Dl = 250 * Mpc;
DS = DLS + Dl; % 观测者距离光源的距离

t = linspace(-10, 0, 10000);

%% 定义信号波形并画出信号
Thetat = eta * t / 5 * M;

xt = Thetat.^(-1/4) / 4;
phit = -Thetat.^(5/8) / eta;

ht = -8 * sqrt(pi / 5) * (mu / DS) .* exp(-2i * phit) .* xt;
ht = ht * 1e26;
% 绘制信号
figure;
plot(t, real(ht), 'b');
xlabel('Time [s]');
ylabel('Strain');
title('0.5PN GW');
grid on;
legend('GW waveform');

%% 定义高斯噪声并绘制
mean_noise = 0; % 噪声的均值
std_dev = 1e-21; % 噪声的标准差

% 生成高斯噪声
gaussian_noise = mean_noise + std_dev * randn(size(t));

% 绘制噪声
figure;
plot(t, gaussian_noise);
xlabel('Time [s]');
ylabel('Amplitude');
title('Gaussian Noise');
grid on;

%% 获取总数据并绘制
data = ht + gaussian_noise;

figure;
plot(t, gaussian_noise, 'Color', [255/255, 192/255, 76/255]);
hold on;
plot(t, real(ht), 'LineWidth', 1.5, 'Color', [31/255, 119/255, 180/255]);
xlabel('Time [s]');
ylabel('Amplitude');
title('DATA');
grid on;

%% 定义透镜因子并绘制透镜化的波形
z = 1.5;
ML = 100 * Msun;
Mlz = ML * (1 + 0.5); % 红移后的透镜质量
epsilon = 0.25 * 1e6; 
epsilon0 = sqrt((4 * G / c^2) * (DLS / (DS * Dl))); % 透镜的爱因斯坦半径
y = epsilon * Dl / (epsilon0 * DS);
beta = sqrt(y.^2 + 4);
miuAdd = 1/2 + (y.^2 / (2 * y * beta));
miuSub = 1/2 - (y.^2 / (2 * y * beta));
deltaTD = 4 * Mlz * (y * beta / 2 + log((beta + y) / (beta - y)));

% 频率数组
f = linspace(0, 1/(2*mean(diff(t))), length(t)/2+1);

% 计算 F(f)
Ff = abs(miuAdd).^(1/2) - abs(miuSub).^(1/2) .* exp(2 * pi * 1i * f * deltaTD);

% 对信号波形进行傅里叶变换
htfft = fft(ht);

% 获得透镜化的波形
hlenf = htfft(1:length(f)) .* Ff;

% 将这个波形转换回时域上
hlent = ifft([hlenf, conj(flip(hlenf(2:end-1)))]);

% 绘制透镜化的波形
figure;
plot(t, real(hlent), 'r', 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('Amplitude');
title('Lensed Signal');
grid on;




