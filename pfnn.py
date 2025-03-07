import numpy as np
import theano
import theano.tensor as T

import math
import sys
sys.path.append('/home/rootai/Desktop/Code/main/pfnn-py-repo/nn')
sys.path.append('/home/rootai/Desktop/Code/main/pfnn-py-repo')

from skeletondef import Choose as skd
from Layer import Layer
from HiddenLayer import HiddenLayer
from BiasLayer import BiasLayer
from DropoutLayer import DropoutLayer
from ActivationLayer import ActivationLayer
from AdamTrainer import AdamTrainer

pfnn_dir = '/home/rootai/Desktop/Code/main/pfnn-py-repo/demo/network/pfnn'

class PhaseFunctionedNetwork(Layer):
    def __init__(self, rng=np.random.RandomState(23456), dropout=0.7, mode='train'):
        """
        初始化PFNN网络
        
        参数:
            mode: 'train' 用于训练, 'predict' 用于预测
        """
        
        self.mode = mode
        self.nslices = 4
        
        if mode == 'train':
            
            """ Load Database """
            
            database = np.load(skd.DATABASE_NAME)
            X = database['Xun'].astype(theano.config.floatX)
            Y = database['Yun'].astype(theano.config.floatX)

            print(X.shape, Y.shape)

            """ Calculate Mean and Std """

            Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
            Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)

            j = skd.JOINT_NUM
            w = ((skd.WINDOW*2)//10)

            Xstd[w*0:w* 1] = Xstd[w*0:w* 1].mean() # Trajectory Past Positions
            Xstd[w*1:w* 2] = Xstd[w*1:w* 2].mean() # Trajectory Future Positions
            Xstd[w*2:w* 3] = Xstd[w*2:w* 3].mean() # Trajectory Past Directions
            Xstd[w*3:w* 4] = Xstd[w*3:w* 4].mean() # Trajectory Future Directions
            Xstd[w*4:w*10] = Xstd[w*4:w*10].mean() # Trajectory Gait

            """ Mask Out Unused Joints in Input """

            joint_weights = np.array(skd.JOINT_WEIGHTS).repeat(3)

            Xstd[w*10+j*3*0:w*10+j*3*1] = Xstd[w*10+j*3*0:w*10+j*3*1].mean() / (joint_weights * 0.1) # Pos
            Xstd[w*10+j*3*1:w*10+j*3*2] = Xstd[w*10+j*3*1:w*10+j*3*2].mean() / (joint_weights * 0.1) # Vel
            Xstd[w*10+j*3*2:          ] = Xstd[w*10+j*3*2:          ].mean() # Terrain

            Ystd[0:2] = Ystd[0:2].mean() # Translational Velocity
            Ystd[2:3] = Ystd[2:3].mean() # Rotational Velocity
            Ystd[3:4] = Ystd[3:4].mean() # Change in Phase
            Ystd[4:8] = Ystd[4:8].mean() # Contacts

            Ystd[8+w*0:8+w*1] = Ystd[8+w*0:8+w*1].mean() # Trajectory Future Positions
            Ystd[8+w*1:8+w*2] = Ystd[8+w*1:8+w*2].mean() # Trajectory Future Directions

            Ystd[8+w*2+j*3*0:8+w*2+j*3*1] = Ystd[8+w*2+j*3*0:8+w*2+j*3*1].mean() # Pos
            Ystd[8+w*2+j*3*1:8+w*2+j*3*2] = Ystd[8+w*2+j*3*1:8+w*2+j*3*2].mean() # Vel
            Ystd[8+w*2+j*3*2:8+w*2+j*3*3] = Ystd[8+w*2+j*3*2:8+w*2+j*3*3].mean() # Rot

            """ Save Normalizers Mean / Std / Min / Max """

            Xmean.astype(np.float32).tofile(f'{pfnn_dir}/Xmean.bin')
            Ymean.astype(np.float32).tofile(f'{pfnn_dir}/Ymean.bin')
            Xstd.astype(np.float32).tofile(f'{pfnn_dir}/Xstd.bin')
            Ystd.astype(np.float32).tofile(f'{pfnn_dir}/Ystd.bin')

            """ Normalize Data """

            self.X = (X - Xmean) / Xstd
            self.Y = (Y - Ymean) / Ystd
            self.P = database['Pun'].astype(theano.config.floatX)
            
            """ Construct Network """
            
            input_shape = self.X.shape[1]+1
            output_shape = self.Y.shape[1]
            print(f"PhaseFunctionedNetwork: {input_shape-1}, {output_shape}")
            
            # 训练模式使用Theano层
            self.dropout0 = DropoutLayer(dropout, rng=rng)
            self.dropout1 = DropoutLayer(dropout, rng=rng)
            self.dropout2 = DropoutLayer(dropout, rng=rng)
            self.activation = ActivationLayer('ELU')
            
            self.W0 = HiddenLayer((self.nslices, 512, input_shape-1), rng=rng, gamma=0.01)
            self.W1 = HiddenLayer((self.nslices, 512, 512), rng=rng, gamma=0.01)
            self.W2 = HiddenLayer((self.nslices, output_shape, 512), rng=rng, gamma=0.01)
        
            self.b0 = BiasLayer((self.nslices, 512))
            self.b1 = BiasLayer((self.nslices, 512))
            self.b2 = BiasLayer((self.nslices, output_shape))

            self.layers = [
                self.W0, self.W1, self.W2,
                self.b0, self.b1, self.b2]

            self.params = sum([layer.params for layer in self.layers], [])
            
        else:
            # 预测模式加载预计算的权重
            self.load_weights()
            self.load_normalizers()

    def load_weights(self):
        """加载预计算的权重"""
        self.W0 = []
        self.W1 = []
        self.W2 = []
        self.b0 = []
        self.b1 = []
        self.b2 = []
        
        # # Binary files
        # for i in range(50):
        #     self.W0.append(np.fromfile(f'{pfnn_dir}/W0_{i:03d}.bin', dtype=np.float32))
        #     self.W1.append(np.fromfile(f'{pfnn_dir}/W1_{i:03d}.bin', dtype=np.float32))
        #     self.W2.append(np.fromfile(f'{pfnn_dir}/W2_{i:03d}.bin', dtype=np.float32))
        #     self.b0.append(np.fromfile(f'{pfnn_dir}/b0_{i:03d}.bin', dtype=np.float32))
        #     self.b1.append(np.fromfile(f'{pfnn_dir}/b1_{i:03d}.bin', dtype=np.float32))
        #     self.b2.append(np.fromfile(f'{pfnn_dir}/b2_{i:03d}.bin', dtype=np.float32))
        # # 重塑权重矩阵
        # input_dim = self.W0[0].size // 512
        # output_dim = self.W2[0].size // 512
        # print("input_dim: ", input_dim, "output_dim: ", output_dim)
        # for i in range(50):
        #     self.W0[i] = self.W0[i].reshape(512, input_dim)
        #     self.W1[i] = self.W1[i].reshape(512, 512)
        #     self.W2[i] = self.W2[i].reshape(output_dim, 512)
        #     self.b0[i] = self.b0[i].reshape(512)
        #     self.b1[i] = self.b1[i].reshape(512)
        #     self.b2[i] = self.b2[i].reshape(output_dim)
        
        # Npy files
        for i in range(50):
            self.W0.append(np.load(f'{pfnn_dir}/W0_{i:03d}.npy'))
            self.W1.append(np.load(f'{pfnn_dir}/W1_{i:03d}.npy'))
            self.W2.append(np.load(f'{pfnn_dir}/W2_{i:03d}.npy'))
            self.b0.append(np.load(f'{pfnn_dir}/b0_{i:03d}.npy'))
            self.b1.append(np.load(f'{pfnn_dir}/b1_{i:03d}.npy'))
            self.b2.append(np.load(f'{pfnn_dir}/b2_{i:03d}.npy'))
    
    def load_normalizers(self):
        """加载标准化参数"""
        self.Xmean = np.fromfile(f'{pfnn_dir}/Xmean.bin', dtype=np.float32)
        self.Xstd = np.fromfile(f'{pfnn_dir}/Xstd.bin', dtype=np.float32)
        self.Ymean = np.fromfile(f'{pfnn_dir}/Ymean.bin', dtype=np.float32)
        self.Ystd = np.fromfile(f'{pfnn_dir}/Ystd.bin', dtype=np.float32)
        

    def save_network(self):
        """保存网络权重到文件"""
        if self.mode != 'train':
            raise Exception("只能在训练模式下保存网络")
            
        W0n = self.W0.W.get_value()
        W1n = self.W1.W.get_value()
        W2n = self.W2.W.get_value()
        b0n = self.b0.b.get_value()
        b1n = self.b1.b.get_value()
        b2n = self.b2.b.get_value()
        
        for i in range(50):
            pscale = self.nslices*(float(i)/50)
            pamount = pscale % 1.0
            
            pindex_1 = int(pscale) % self.nslices
            pindex_0 = (pindex_1-1) % self.nslices
            pindex_2 = (pindex_1+1) % self.nslices
            pindex_3 = (pindex_1+2) % self.nslices
            
            def cubic(y0, y1, y2, y3, mu):
                return (
                    (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
                    (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
                    (-0.5*y0+0.5*y2)*mu +
                    (y1))
            
            W0 = cubic(W0n[pindex_0], W0n[pindex_1], W0n[pindex_2], W0n[pindex_3], pamount)
            W1 = cubic(W1n[pindex_0], W1n[pindex_1], W1n[pindex_2], W1n[pindex_3], pamount)
            W2 = cubic(W2n[pindex_0], W2n[pindex_1], W2n[pindex_2], W2n[pindex_3], pamount)
            b0 = cubic(b0n[pindex_0], b0n[pindex_1], b0n[pindex_2], b0n[pindex_3], pamount)
            b1 = cubic(b1n[pindex_0], b1n[pindex_1], b1n[pindex_2], b1n[pindex_3], pamount)
            b2 = cubic(b2n[pindex_0], b2n[pindex_1], b2n[pindex_2], b2n[pindex_3], pamount)
            
            # Binary files
            W0.astype(np.float32).tofile(f'{pfnn_dir}/W0_{i:03d}.bin')
            W1.astype(np.float32).tofile(f'{pfnn_dir}/W1_{i:03d}.bin')
            W2.astype(np.float32).tofile(f'{pfnn_dir}/W2_{i:03d}.bin')
            b0.astype(np.float32).tofile(f'{pfnn_dir}/b0_{i:03d}.bin')
            b1.astype(np.float32).tofile(f'{pfnn_dir}/b1_{i:03d}.bin')
            b2.astype(np.float32).tofile(f'{pfnn_dir}/b2_{i:03d}.bin')
            
            # Npy files
            # Saving to npy files without converting data type will not have errors when writing and reading
            np.save(f'{pfnn_dir}/W0_{i:03d}.npy', W0)
            np.save(f'{pfnn_dir}/W1_{i:03d}.npy', W1)
            np.save(f'{pfnn_dir}/W2_{i:03d}.npy', W2)
            np.save(f'{pfnn_dir}/b0_{i:03d}.npy', b0)
            np.save(f'{pfnn_dir}/b1_{i:03d}.npy', b1)
            np.save(f'{pfnn_dir}/b2_{i:03d}.npy', b2)
            

        # region 测试读写文件
        # # Binary files
        # W0_read_bin = np.fromfile(f'{pfnn_dir}/W0_049.bin', dtype=np.float32)
        # W1_read_bin = np.fromfile(f'{pfnn_dir}/W1_049.bin', dtype=np.float32)
        # W2_read_bin = np.fromfile(f'{pfnn_dir}/W2_049.bin', dtype=np.float32)
        # b0_read_bin = np.fromfile(f'{pfnn_dir}/b0_049.bin', dtype=np.float32)
        # b1_read_bin = np.fromfile(f'{pfnn_dir}/b1_049.bin', dtype=np.float32)
        # b2_read_bin = np.fromfile(f'{pfnn_dir}/b2_049.bin', dtype=np.float32)
         
        # input_dim_bin = W0_read_bin.size // 512
        # output_dim_bin = W2_read_bin.size // 512
        # W0_reshape_bin = W0_read_bin.reshape(512, input_dim_bin)
        # W1_reshape_bin = W1_read_bin.reshape(512, 512)
        # W2_reshape_bin = W2_read_bin.reshape(output_dim_bin, 512)
        # b0_reshape_bin = b0_read_bin.reshape(512)
        # b1_reshape_bin = b1_read_bin.reshape(512)
        # b2_reshape_bin = b2_read_bin.reshape(output_dim_bin)
        # print("\n\n BINARY files")
        # print(f'W0_049, write {len(W0.astype(np.float32))}, read orig {len(W0_read_bin)}, read reshaped {len(W0_reshape_bin)}')
        # print(f'W1_049, write {len(W1.astype(np.float32))}, read orig {len(W1_read_bin)}, read reshaped {len(W1_reshape_bin)}')
        # print(f'W2_049, write {len(W2.astype(np.float32))}, read orig {len(W2_read_bin)}, read reshaped {len(W2_reshape_bin)}')
        # print(f'b0_049, write {len(b0.astype(np.float32))}, read orig {len(b0_read_bin)}, read reshaped {len(b0_reshape_bin)}')
        # print(f'b1_049, write {len(b1.astype(np.float32))}, read orig {len(b1_read_bin)}, read reshaped {len(b1_reshape_bin)}')
        # print(f'b2_049, write {len(b2.astype(np.float32))}, read orig {len(b2_read_bin)}, read reshaped {len(b2_reshape_bin)}')
        # # print("Minus read: ")
        # # print(W0 - W0_read_bin)
        # print("Minus reshape: ")
        # print(W0 - W0_reshape_bin)
        
                
        # # Npy files
        # W0_read_npy = np.load(f'{pfnn_dir}/W0_049.npy').astype(np.float32)
        # W1_read_npy = np.load(f'{pfnn_dir}/W1_049.npy').astype(np.float32)
        # W2_read_npy = np.load(f'{pfnn_dir}/W2_049.npy').astype(np.float32)
        # b0_read_npy = np.load(f'{pfnn_dir}/b0_049.npy').astype(np.float32)
        # b1_read_npy = np.load(f'{pfnn_dir}/b1_049.npy').astype(np.float32)
        # b2_read_npy = np.load(f'{pfnn_dir}/b2_049.npy').astype(np.float32)

        # input_dim_npy = W0_read_npy.size // 512
        # output_dim_npy = W2_read_npy.size // 512
        # W0_reshape_npy = W0_read_npy.reshape(512, input_dim_npy)
        # W1_reshape_npy = W1_read_npy.reshape(512, 512)
        # W2_reshape_npy = W2_read_npy.reshape(output_dim_npy, 512)
        # b0_reshape_npy = b0_read_npy.reshape(512)
        # b1_reshape_npy = b1_read_npy.reshape(512)
        # b2_reshape_npy = b2_read_npy.reshape(output_dim_npy)
        # print("\n\n NPY files")
        # print(f'W0_049, write {len(W0.astype(np.float32))}, read orig {len(W0_read_npy)}, read reshaped {len(W0_reshape_npy)}')
        # print(f'W1_049, write {len(W1.astype(np.float32))}, read orig {len(W1_read_npy)}, read reshaped {len(W1_reshape_npy)}')
        # print(f'W2_049, write {len(W2.astype(np.float32))}, read orig {len(W2_read_npy)}, read reshaped {len(W2_reshape_npy)}')
        # print(f'b0_049, write {len(b0.astype(np.float32))}, read orig {len(b0_read_npy)}, read reshaped {len(b0_reshape_npy)}')
        # print(f'b1_049, write {len(b1.astype(np.float32))}, read orig {len(b1_read_npy)}, read reshaped {len(b1_reshape_npy)}')
        # print(f'b2_049, write {len(b2.astype(np.float32))}, read orig {len(b2_read_npy)}, read reshaped {len(b2_reshape_npy)}')
        # print("Minus read: ")
        # print(W0 - W0_read_npy)
        # print("Minus reshape: ")
        # print(W0 - W0_reshape_npy)
        # endregion
        
        
    def __call__(self, input):
        """训练时的前向传播"""
        if self.mode != 'train':
            raise Exception("只能在训练模式下调用")
            
        pscale = self.nslices * input[:,-1]
        pamount = pscale % 1.0
        
        pindex_1 = T.cast(pscale, 'int32') % self.nslices
        pindex_0 = (pindex_1-1) % self.nslices
        pindex_2 = (pindex_1+1) % self.nslices
        pindex_3 = (pindex_1+2) % self.nslices
        
        Wamount = pamount.dimshuffle(0, 'x', 'x')
        bamount = pamount.dimshuffle(0, 'x')
        
        def cubic(y0, y1, y2, y3, mu):
            return (
                (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
                (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
                (-0.5*y0+0.5*y2)*mu +
                (y1))
        
        W0 = cubic(self.W0.W[pindex_0], self.W0.W[pindex_1], self.W0.W[pindex_2], self.W0.W[pindex_3], Wamount)
        W1 = cubic(self.W1.W[pindex_0], self.W1.W[pindex_1], self.W1.W[pindex_2], self.W1.W[pindex_3], Wamount)
        W2 = cubic(self.W2.W[pindex_0], self.W2.W[pindex_1], self.W2.W[pindex_2], self.W2.W[pindex_3], Wamount)
        
        b0 = cubic(self.b0.b[pindex_0], self.b0.b[pindex_1], self.b0.b[pindex_2], self.b0.b[pindex_3], bamount)
        b1 = cubic(self.b1.b[pindex_0], self.b1.b[pindex_1], self.b1.b[pindex_2], self.b1.b[pindex_3], bamount)
        b2 = cubic(self.b2.b[pindex_0], self.b2.b[pindex_1], self.b2.b[pindex_2], self.b2.b[pindex_3], bamount)
        
        H0 = input[:,:-1]
        H1 = self.activation(T.batched_dot(W0, self.dropout0(H0)) + b0)
        H2 = self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1)
        H3 =                 T.batched_dot(W2, self.dropout2(H2)) + b2
        
        return H3
    

    def predict(self, X, P):
        """预测模式下的前向传播"""
        if self.mode != 'predict':
            raise Exception("只能在预测模式下调用predict")
            
        # 标准化输入
        Xp = (X - self.Xmean) / self.Xstd
        
        # 计算相位索引
        pindex = int(P / (2 * math.pi) * 50)
        
        # 使用预计算的权重
        W0p = self.W0[pindex]
        W1p = self.W1[pindex] 
        W2p = self.W2[pindex]
        b0p = self.b0[pindex]
        b1p = self.b1[pindex]
        b2p = self.b2[pindex]

        # 前向传播
        def elu(x):
            return np.where(x > 0, x, np.exp(x) - 1)
            
        H0 = elu(np.dot(W0p, Xp) + b0p)
        H1 = elu(np.dot(W1p, H0) + b1p)
        Yp = np.dot(W2p, H1) + b2p
        
        # 反标准化输出
        Y = (Yp * self.Ystd) + self.Ymean
        
        return Y

    def cost(self, input):
        input = input[:,:-1]
        costs = 0
        for layer in self.layers:
            costs += layer.cost(input)
            input = layer(input)
        return costs / len(self.layers)
    
    def save(self, database, prefix=''):
        for li, layer in enumerate(self.layers):
            layer.save(database, '%sL%03i_' % (prefix, li))
        
    def load(self, database, prefix=''):
        for li, layer in enumerate(self.layers):
            layer.load(database, '%sL%03i_' % (prefix, li)) 