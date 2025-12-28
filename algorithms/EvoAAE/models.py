import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class ConvEncoder(nn.Module):
    """卷积编码器，使用1D卷积处理时间序列"""
    def __init__(self, input_dim, latent_dim, conv_channels=[32, 64], kernel_sizes=[3, 3]):
        super(ConvEncoder, self).__init__()
        
        # 构建卷积层
        layers = []
        in_channels = input_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels) if i < len(conv_channels)-1 else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2) if i < len(conv_channels)-1 else nn.Identity()
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 计算卷积后的特征维度
        self.conv_output_dim = conv_channels[-1]
        
        # 均值和对数方差层
        self.fc_mean = nn.Linear(self.conv_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_dim, latent_dim)
    
    def forward(self, x):
        # x形状: (batch_size, seq_len, input_dim) -> 转置为 (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # 卷积编码
        conv_out = self.conv_layers(x)
        
        # 全局平均池化
        pooled = torch.mean(conv_out, dim=2)
        
        # 计算均值和方差
        z_mean = self.fc_mean(pooled)
        z_logvar = self.fc_logvar(pooled)
        
        return z_mean, z_logvar

class ConvDecoder(nn.Module):
    """卷积解码器，使用转置卷积重建时间序列"""
    def __init__(self, latent_dim, output_dim, output_seq_len, conv_channels=[64, 32], kernel_sizes=[3, 3]):
        super(ConvDecoder, self).__init__()
        
        self.output_seq_len = output_seq_len
        self.output_dim = output_dim
        # 解码阶段除首层外每层都上采样一次，因此上采样次数 = 总层数 - 2（首层和最后一层不放大）
        self.num_upsamples = max(len(conv_channels) - 2, 0)
        
        # 初始全连接层，将潜在向量扩展到卷积输入
        init_seq_len = max(self.output_seq_len // (2**self.num_upsamples), 1)
        self.init_fc = nn.Linear(latent_dim, conv_channels[0] * init_seq_len)
        
        # 构建转置卷积层
        layers = []
        in_channels = conv_channels[0]
        
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels[1:], kernel_sizes[1:])):
            layers.extend([
                nn.Upsample(scale_factor=2) if i > 0 else nn.Identity(),
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels) if i < len(conv_channels)-2 else nn.Identity(),
                nn.ReLU(inplace=True) if i < len(conv_channels)-2 else nn.Identity()
            ])
            in_channels = out_channels
        
        # 最后一层转置卷积，输出到原始维度
        layers.extend([
            nn.ConvTranspose1d(in_channels, output_dim, kernel_sizes[-1], padding=kernel_sizes[-1]//2),
            # 添加Tanh激活函数，将输出限制在[-1, 1]范围
            nn.Tanh()
        ])
        
        self.deconv_layers = nn.Sequential(*layers)
    
    def forward(self, z):
        batch_size = z.size(0)
        
        # 初始全连接
        x = self.init_fc(z)
        
        # 重塑为卷积输入格式
        seq_len = max(self.output_seq_len // (2**self.num_upsamples), 1)
        x = x.view(batch_size, -1, seq_len)
        
        # 转置卷积解码
        x_recon = self.deconv_layers(x)
        # 调整回目标序列长度，避免卷积与上采样造成长度漂移
        if x_recon.size(2) != self.output_seq_len:
            x_recon = F.interpolate(
                x_recon,
                size=self.output_seq_len,
                mode='linear',
                align_corners=False
            )
        
        # 转置回原始格式: (batch_size, seq_len, output_dim)
        x_recon = x_recon.transpose(1, 2)
        
        return x_recon

class ConvDiscriminator(nn.Module):
    """卷积判别器"""
    def __init__(self, input_dim, conv_channels=[32, 64, 128], kernel_sizes=[3, 3, 3]):
        super(ConvDiscriminator, self).__init__()
        
        layers = []
        in_channels = input_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels) if i > 0 else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool1d(2) if i < len(conv_channels)-1 else nn.Identity()
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 计算卷积后的特征维度
        self.conv_output_dim = conv_channels[-1]
        
        # 分类头
        # 输出原始logits，配合 BCEWithLogitsLoss；不要提前Sigmoid
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_dim, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x形状: (batch_size, seq_len, input_dim) -> 转置为 (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # 卷积特征提取，序列过短时跳过会让长度归零的池化
        for layer in self.conv_layers:
            if isinstance(layer, nn.MaxPool1d):
                k = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
                if x.size(2) < k:
                    continue
            x = layer(x)
        conv_out = x
        
        # 全局平均池化
        pooled = torch.mean(conv_out, dim=2)
        
        # 分类
        output = self.fc(pooled)
        
        return output

class AdversarialAutoencoderWithDualDiscriminator(nn.Module):
    """
    论文中的对抗自编码器模型（图3）
    包含编码器、解码器和两个判别器
    """
    def __init__(self, config, device):
        super(AdversarialAutoencoderWithDualDiscriminator, self).__init__()
        
        # 从配置中获取参数
        self.input_dim = config['input_dim']
        self.seq_len = config['seq_len']
        self.latent_dim = config['latent_dim']
        self.conv_channels = config.get('conv_channels', [32, 64])
        self.kernel_sizes = config.get('kernel_sizes', [3, 3])
        self.kl_beta = config.get('kl_beta', 1.0)
        self.adv_weight_latent = config.get('adv_weight_latent', 1.0)
        self.adv_weight_data = config.get('adv_weight_data', 1.0)
        self.device = device
        
        # 构建四个网络（论文图3）
        self.encoder = ConvEncoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            conv_channels=self.conv_channels,
            kernel_sizes=self.kernel_sizes
        )
        
        self.decoder = ConvDecoder(
            latent_dim=self.latent_dim,
            output_dim=self.input_dim,
            output_seq_len=self.seq_len,
            conv_channels=list(reversed(self.conv_channels)),
            kernel_sizes=list(reversed(self.kernel_sizes))
        )
        
        self.latent_discriminator = ConvDiscriminator(
            input_dim=self.latent_dim,
            conv_channels=[16, 32],
            kernel_sizes=[3, 3]
        )
        
        self.data_discriminator = ConvDiscriminator(
            input_dim=self.input_dim,
            conv_channels=self.conv_channels,
            kernel_sizes=self.kernel_sizes
        )
        
        # 损失权重
        self.recon_criterion = nn.MSELoss(reduction='mean')
        self.adv_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        
        # 优化器
        self.enc_dec_optimizer = None
        self.latent_disc_optimizer = None
        
        # 权重初始化
        self._initialize_weights()
        self.data_disc_optimizer = None
    
    def _initialize_weights(self):
        """使用Xavier均匀初始化来稳定权重"""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # 降低增益使初始梯度更小
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
    def to_device(self):
        """移动模型到设备"""
        self.to(self.device)
        return self
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """前向传播"""
        # 编码
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)
        
        # 解码
        x_recon = self.decoder(z)
        
        return x_recon, z_mean, z_logvar, z
    
    def set_learning_rate(self, lr):
        """动态设置学习率"""
        for param_group in self.enc_dec_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.latent_disc_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.data_disc_optimizer.param_groups:
            param_group['lr'] = lr
    
    def compile_optimizers(self, enc_dec_lr=0.001, latent_disc_lr=0.0005, data_disc_lr=0.0005):
        """初始化优化器，存储初始学习率以支持预热"""
        self.base_lr = enc_dec_lr
        
        self.enc_dec_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=enc_dec_lr * 0.1,  # 初始化为较小的学习率，然后逐步增加
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.latent_disc_optimizer = optim.Adam(
            self.latent_discriminator.parameters(),
            lr=latent_disc_lr * 0.1,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.data_disc_optimizer = optim.Adam(
            self.data_discriminator.parameters(),
            lr=data_disc_lr * 0.1,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def train_step(self, x, train_discriminators=True):
        """单步训练（论文Algorithm 3）"""
        batch_size = x.size(0)
        # light label smoothing to stabilize GAN
        real_labels = torch.full((batch_size, 1), 0.95, device=self.device)
        fake_labels = torch.full((batch_size, 1), 0.05, device=self.device)
        
        # ========== 1. 训练编码器-解码器（重构+KL损失） ==========
        self.enc_dec_optimizer.zero_grad()
        
        # 前向传播
        x_recon, z_mean, z_logvar, z = self.forward(x)
        
        # 计算重构损失
        recon_loss = self.recon_criterion(x_recon, x)
        
        # 计算KL散度损失
        kl_loss = -0.5 * torch.mean(torch.sum(1 + z_logvar - z_mean**2 - torch.exp(z_logvar), dim=1))
        
        # VAE总损失
        vae_loss = recon_loss + self.kl_beta * kl_loss
        vae_loss.backward(retain_graph=True)
        
        # 梯度裁剪，防止梯度爆炸（严格设置）
        enc_grad_norm = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        dec_grad_norm = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
        
        # 梯度监控
        if enc_grad_norm > 10.0 or dec_grad_norm > 10.0:
            print(f"警告: 编码器梯度范数={enc_grad_norm:.2f}, 解码器梯度范数={dec_grad_norm:.2f}")
        
        self.enc_dec_optimizer.step()
        
        if train_discriminators:
            # ========== 2. 训练潜在空间判别器 ==========
            self.latent_disc_optimizer.zero_grad()
            
            # 真实潜在向量（标准正态分布）
            z_real = torch.randn(batch_size, self.latent_dim).to(self.device)
            
            # 生成潜在向量（当前编码器的输出）
            with torch.no_grad():
                _, _, _, z_fake = self.forward(x)
            
            # 判别器输出（logits）
            d_real = self.latent_discriminator(z_real.unsqueeze(1))
            d_fake = self.latent_discriminator(z_fake.unsqueeze(1))
            
            # 计算对抗损失（BCEWithLogitsLoss直接接收logits）
            ld_loss_real = self.adv_criterion(d_real, real_labels)
            ld_loss_fake = self.adv_criterion(d_fake, fake_labels)
            ld_loss = 0.5 * (ld_loss_real + ld_loss_fake)
            
            ld_loss.backward()
            
            # 梯度裁剪（严格设置）
            ld_grad_norm = torch.nn.utils.clip_grad_norm_(self.latent_discriminator.parameters(), max_norm=1.0)
            if ld_grad_norm > 10.0:
                print(f"警告: 潜空间判别器梯度范数={ld_grad_norm:.2f}")
            
            self.latent_disc_optimizer.step()
            
            # ========== 3. 训练数据空间判别器 ==========
            self.data_disc_optimizer.zero_grad()
            
            # 真实数据
            x_real = x
            
            # 生成数据（当前解码器的输出）
            with torch.no_grad():
                x_fake, _, _, _ = self.forward(x)
            
            # 判别器输出（logits）
            d_real_data = self.data_discriminator(x_real)
            d_fake_data = self.data_discriminator(x_fake)
            
            # 计算对抗损失
            xd_loss_real = self.adv_criterion(d_real_data, real_labels)
            xd_loss_fake = self.adv_criterion(d_fake_data, fake_labels)
            xd_loss = 0.5 * (xd_loss_real + xd_loss_fake)
            
            xd_loss.backward()
            
            # 梯度裁剪（严格设置）
            xd_grad_norm = torch.nn.utils.clip_grad_norm_(self.data_discriminator.parameters(), max_norm=1.0)
            if xd_grad_norm > 10.0:
                print(f"警告: 数据空间判别器梯度范数={xd_grad_norm:.2f}")
            
            self.data_disc_optimizer.step()
        
        # ========== 4. 训练编码器以欺骗判别器 ==========
        self.enc_dec_optimizer.zero_grad()
        
        # 前向传播
        x_recon_adv, _, _, z_adv = self.forward(x)
        
        # 判别器输出（不停止梯度）
        d_latent_adv = self.latent_discriminator(z_adv.unsqueeze(1))
        d_data_adv = self.data_discriminator(x_recon_adv)
        
        # 对抗损失：让判别器认为生成数据是真实的
        adv_loss_latent = self.adv_criterion(d_latent_adv, real_labels)
        adv_loss_data = self.adv_criterion(d_data_adv, real_labels)
        adv_loss = self.adv_weight_latent * adv_loss_latent + self.adv_weight_data * adv_loss_data
        
        adv_loss.backward()
        
        # 梯度裁剪（严格设置）
        enc_adv_grad_norm = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        dec_adv_grad_norm = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
        
        # 梯度监控
        if enc_adv_grad_norm > 10.0 or dec_adv_grad_norm > 10.0:
            print(f"警告（ADV阶段）: 编码器梯度范数={enc_adv_grad_norm:.2f}, 解码器梯度范数={dec_adv_grad_norm:.2f}")
        
        self.enc_dec_optimizer.step()
        
        # 总损失
        total_loss = vae_loss + adv_loss
        
        # 返回损失字典
        losses = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'ld_loss': ld_loss.item() if train_discriminators else 0.0,
            'xd_loss': xd_loss.item() if train_discriminators else 0.0,
            'adv_loss': adv_loss.item()
        }
        
        return losses
    
    def fit(self, X_train, epochs=50, batch_size=32, validation_data=None, verbose=1):
        """训练模型"""
        self.train()
        
        # 转换为张量
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.FloatTensor(X_train).to(self.device)
        else:
            X_train = X_train.to(self.device)
        
        # 检查数据中是否有NaN或Inf
        if torch.any(~torch.isfinite(X_train)):
            print("警告：输入数据中包含NaN或Inf值，进行清理...")
            X_train = torch.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 检查数据范围
        data_min = torch.min(X_train).item()
        data_max = torch.max(X_train).item()
        data_mean = torch.mean(X_train).item()
        data_std = torch.std(X_train).item()
        print(f"训练数据统计: min={data_min:.4f}, max={data_max:.4f}, mean={data_mean:.4f}, std={data_std:.4f}")
        
        # 数据加载器
        train_dataset = TensorDataset(X_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # 验证数据
        if validation_data is not None:
            if not isinstance(validation_data, torch.Tensor):
                X_val = torch.FloatTensor(validation_data).to(self.device)
            else:
                X_val = validation_data.to(self.device)
        
        # 历史记录
        history = {
            'loss': [], 'recon_loss': [], 'kl_loss': [], 
            'ld_loss': [], 'xd_loss': [], 'adv_loss': [],
            'val_loss': [], 'val_recon_loss': [], 'val_kl_loss': [],
            'val_ld_loss': [], 'val_xd_loss': [], 'val_adv_loss': []
        }
        
        # 学习率预热配置（保持不变）
        warmup_epochs = min(5, max(1, epochs // 10))
        
        for epoch in range(epochs):
            # 学习率预热：逐步从0.1倍增加到1倍
            if epoch < warmup_epochs:
                warmup_progress = (epoch + 1) / warmup_epochs
                current_lr = self.base_lr * warmup_progress * 0.1
                self.set_learning_rate(current_lr)
                if verbose:
                    print(f"[预热阶段] Epoch {epoch+1} - 学习率: {current_lr:.6f}")
            else:
                # 预热完成后，设置为正常学习率
                if epoch == warmup_epochs:
                    self.set_learning_rate(self.base_lr)
            
            epoch_losses = {
                'loss': [], 'recon_loss': [], 'kl_loss': [],
                'ld_loss': [], 'xd_loss': [], 'adv_loss': []
            }
            
            # 每个batch都训练判别器，增强对抗信号
            train_discriminators = True
            
            for batch_idx, batch in enumerate(train_loader):
                X_batch = batch[0]
                losses = self.train_step(X_batch, train_discriminators=train_discriminators)
                
                # 检查损失是否异常
                total_loss = losses['total_loss']
                if total_loss > 1e6:
                    print(f"警告：epoch {epoch+1}, batch {batch_idx+1} 损失异常大: {total_loss:.2e}")
                    print(f"  各分量: recon_loss={losses['recon_loss']:.4f}, kl_loss={losses['kl_loss']:.4f}")
                    print(f"  adv_loss={losses['adv_loss']:.4f}, ld_loss={losses['ld_loss']:.4f}")
                
                for key in epoch_losses:
                    if key == 'loss':
                        epoch_losses[key].append(losses['total_loss'])
                    else:
                        epoch_losses[key].append(losses[key])
            
            # 计算平均损失
            for key in epoch_losses:
                avg_loss = np.mean(epoch_losses[key])
                history[key].append(avg_loss)
            
            # 验证
            if validation_data is not None:
                val_losses = self.evaluate(X_val)
                for key, value in val_losses.items():
                    history[f'val_{key}'].append(value)
            
            # 打印进度
            if verbose and (epoch + 1) % 10 == 0:
                log_msg = f"Epoch {epoch+1}/{epochs} - loss: {history['loss'][-1]:.4f}"
                if validation_data is not None:
                    log_msg += f" - val_loss: {history['val_loss'][-1]:.4f}"
                print(log_msg)
        
        return history
    
    def evaluate(self, X):
        """评估模型"""
        self.eval()
        with torch.no_grad():
            x_recon, z_mean, z_logvar, z = self.forward(X)
            
            # 计算损失
            recon_loss = self.recon_criterion(x_recon, X)
            kl_loss = -0.5 * torch.mean(torch.sum(1 + z_logvar - z_mean**2 - torch.exp(z_logvar), dim=1))
            
            # 判别器损失
            batch_size = X.size(0)
            z_real = torch.randn(batch_size, self.latent_dim).to(self.device)
            # label smoothing in eval path as well
            real_labels = torch.full((batch_size, 1), 0.95, device=self.device)
            fake_labels = torch.full((batch_size, 1), 0.05, device=self.device)
            
            # 潜在判别器损失
            ld_real = self.latent_discriminator(z_real.unsqueeze(1))
            ld_fake = self.latent_discriminator(z.unsqueeze(1))
            ld_loss = 0.5 * (self.adv_criterion(ld_real, real_labels) + self.adv_criterion(ld_fake, fake_labels))
            
            # 数据判别器损失
            xd_real = self.data_discriminator(X)
            xd_fake = self.data_discriminator(x_recon)
            xd_loss = 0.5 * (self.adv_criterion(xd_real, real_labels) + self.adv_criterion(xd_fake, fake_labels))
            
            # 对抗损失
            adv_loss = self.adv_weight_latent * self.adv_criterion(ld_fake, real_labels) + \
                       self.adv_weight_data * self.adv_criterion(xd_fake, real_labels)
            
            total_loss = recon_loss + self.kl_beta * kl_loss + adv_loss
        
        self.train()
        return {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'ld_loss': ld_loss.item(),
            'xd_loss': xd_loss.item(),
            'adv_loss': adv_loss.item()
        }
    
    def compute_reconstruction_error(self, X):
        """计算重构误差（用于异常检测）"""
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X).to(self.device)
            else:
                X = X.to(self.device)
            
            x_recon, _, _, _ = self.forward(X)
            # MSE误差按序列和特征维度平均
            mse = torch.mean((X - x_recon) ** 2, dim=(1, 2))
            
            # 获取MSE值
            mse_np = mse.cpu().numpy()
            
            # 安全检查：移除NaN和Inf值
            mse_np = np.nan_to_num(mse_np, nan=0.0, posinf=1e10, neginf=0.0)
            
            return mse_np