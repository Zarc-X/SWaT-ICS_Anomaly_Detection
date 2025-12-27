import numpy as np

class BinaryPSO:
    """
    二进制粒子群优化器（论文IV.B节）
    优化8类参数：批量大小、学习率、优化器类型、卷积层数、卷积核数、核大小、归一化层类型、激活函数
    """
    def __init__(self, param_options, pop_size=20, max_iter=30, inertia_max=0.9, inertia_min=0.4):
        """
        参数选项格式:
        param_options = {
            'batch_size': [1024, 2048, 4096, 6144],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
            'optimizer': ['adam', 'adamax', 'rmsprop', 'adadelta'],
            'n_layers': [3, 4, 5, 6],
            'n_kernels': [2, 4, 8, 16, 32, 64, 128, 256],
            'kernel_size': [1, 2, 3, 4],
            'normalization': ['batchnorm', 'none'],
            'activation': ['relu', 'sigmoid', 'tanh', 'none']
        }
        """
        self.param_options = param_options
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.inertia_max = inertia_max
        self.inertia_min = inertia_min
        
        # 计算二进制编码长度
        self.param_lengths = {}
        self.total_length = 0
        
        for param_name, options in param_options.items():
            # 计算需要多少位来表示这些选项
            n_options = len(options)
            n_bits = int(np.ceil(np.log2(n_options)))
            
            # 确保二进制位数足够表示所有选项
            while 2**n_bits < n_options:
                n_bits += 1
            
            self.param_lengths[param_name] = n_bits
            self.total_length += n_bits
        
        print(f"总编码长度: {self.total_length} 位")
        for param_name, length in self.param_lengths.items():
            print(f"  {param_name}: {length} 位 ({len(param_options[param_name])} 个选项)")
        
        # 初始化粒子群
        self.swarm = self._initialize_swarm()
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.history = []
    
    def _initialize_swarm(self):
        """初始化粒子群"""
        swarm = []
        
        for _ in range(self.pop_size):
            # 随机生成二进制编码
            binary_string = np.random.randint(0, 2, self.total_length)
            
            # 解码为实际参数
            position = self._decode_binary(binary_string)
            
            # 随机初始化速度
            velocity = np.random.uniform(-1, 1, self.total_length)
            
            swarm.append({
                'binary_position': binary_string.copy(),
                'position': position,
                'velocity': velocity,
                'best_binary_position': binary_string.copy(),
                'best_position': position.copy(),
                'best_fitness': float('inf'),
                'fitness': float('inf')
            })
        
        return swarm
    
    def _decode_binary(self, binary_string):
        """解码二进制字符串为实际参数值（论文图6）"""
        decoded = {}
        start_idx = 0
        
        for param_name, n_bits in self.param_lengths.items():
            # 提取对应位的二进制码
            bits = binary_string[start_idx:start_idx + n_bits]
            
            # 转换为整数索引
            idx = 0
            for bit in bits:
                idx = (idx << 1) | int(bit)
            
            # 确保索引在选项范围内
            options = self.param_options[param_name]
            idx = idx % len(options)
            
            # 获取实际参数值
            decoded[param_name] = options[idx]
            
            start_idx += n_bits
        
        return decoded
    
    def _encode_position(self, position):
        """编码参数位置为二进制字符串"""
        binary_string = np.zeros(self.total_length, dtype=int)
        current_idx = 0
        
        for param_name, value in position.items():
            options = self.param_options[param_name]
            n_bits = self.param_lengths[param_name]
            
            # 查找值在选项中的索引
            try:
                idx = options.index(value)
            except ValueError:
                idx = 0
            
            # 将索引转换为二进制
            for i in range(n_bits):
                bit = (idx >> (n_bits - 1 - i)) & 1
                binary_string[current_idx + i] = bit
            
            current_idx += n_bits
        
        return binary_string
    
    def _sigmoid(self, x):
        """S型函数，用于将速度转换为概率"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def update_velocity_and_position(self, particle, global_best_binary, inertia_weight):
        """更新粒子的速度和位置（论文公式(24)和(26)）"""
        # 更新速度
        r1, r2 = np.random.rand(2)
        
        particle['velocity'] = (
            inertia_weight * particle['velocity'] +
            2.0 * r1 * (particle['best_binary_position'] - particle['binary_position']) +
            2.0 * r2 * (global_best_binary - particle['binary_position'])
        )
        
        # 限制速度范围
        particle['velocity'] = np.clip(particle['velocity'], -6, 6)
        
        # 更新位置（二进制）
        prob = self._sigmoid(particle['velocity'])
        new_binary = (np.random.rand(self.total_length) < prob).astype(int)
        
        # 解码新位置
        particle['binary_position'] = new_binary
        particle['position'] = self._decode_binary(new_binary)
    
    def optimize(self, fitness_function, verbose=True):
        """执行PSO优化"""
        print(f"开始二进制PSO优化，种群大小: {self.pop_size}, 最大迭代次数: {self.max_iter}")
        
        for iteration in range(self.max_iter):
            # 计算惯性权重（线性递减）
            inertia_weight = self.inertia_max - (self.inertia_max - self.inertia_min) * (iteration / self.max_iter)
            
            # 评估每个粒子的适应度
            for particle in self.swarm:
                particle['fitness'] = fitness_function(particle['position'])
                
                # 更新个体最优
                if particle['fitness'] < particle['best_fitness']:
                    particle['best_fitness'] = particle['fitness']
                    particle['best_binary_position'] = particle['binary_position'].copy()
                    particle['best_position'] = particle['position'].copy()
            
            # 更新全局最优
            for particle in self.swarm:
                if particle['fitness'] < self.global_best_fitness:
                    self.global_best_fitness = particle['fitness']
                    self.global_best = particle['position'].copy()
            
            # 记录历史
            self.history.append({
                'iteration': iteration,
                'global_best_fitness': self.global_best_fitness,
                'global_best': self.global_best.copy() if self.global_best else None
            })
            
            # 更新粒子的速度和位置
            global_best_binary = self._encode_position(self.global_best) if self.global_best else np.zeros(self.total_length)
            
            for particle in self.swarm:
                self.update_velocity_and_position(particle, global_best_binary, inertia_weight)
            
            # 打印进度
            if verbose and (iteration + 1) % 5 == 0:
                print(f"迭代 {iteration + 1}/{self.max_iter} - 最佳适应度: {self.global_best_fitness:.6f}")
        
        print(f"优化完成！最佳适应度: {self.global_best_fitness:.6f}")
        print("最佳参数配置:")
        for param_name, value in self.global_best.items():
            print(f"  {param_name}: {value}")
        
        return self.global_best, self.global_best_fitness