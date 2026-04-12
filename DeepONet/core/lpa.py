# core/lpa.py
import numpy as np
import tensorflow as tf
import sympy as sp

def get_Legendre_coefs(order=0, n_panel=10):
    x = sp.symbols('x')
    P = sp.legendre(order, x)
    P_int = sp.integrate(P, x)
    
    inds = np.linspace(-1, 1, n_panel + 1)
    coefs = np.array([float(P_int.subs(x, ind)) for ind in inds], dtype=np.float64)
    
    coefs = coefs[1:] - coefs[:-1]
    coefs *= (2.0 * order + 1.0) / 2.0
    return coefs

class LPA(tf.keras.layers.Layer):
    """
    [Optimized] Vectorized Channel-wise LPA
    - for 루프 제거 -> 행렬 연산으로 속도 향상
    - Bottleneck 구조와 함께 사용 시 효율 극대화
    """
    def __init__(self, order=3, N_p=8, dtype="float32",
                 kernel_regularizer=None, use_softmax=False,
                 project="tanh", name=None):
        super(LPA, self).__init__(name=name)
        self.order = int(order)
        self.N_p = int(N_p)
        self.dtype_ = tf.as_dtype(dtype)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.use_softmax = bool(use_softmax)
        self.project = project

        np_dtype = np.float32 if self.dtype_ == tf.float32 else np.float64
        self.coefs_np = np.array(
            [get_Legendre_coefs(i, self.N_p) for i in range(1, self.order + 1)],
            dtype=np_dtype
        )

    def build(self, input_shape):
        self.n_channels = input_shape[-1]
        
        # [Params] (Channels, N_p)
        self.W_i = self.add_weight(
            "W_i",
            shape=(self.n_channels, self.N_p),
            initializer="glorot_normal",
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=self.dtype_
        )
        
        # Coefs: (Order, N_p)
        self.coefs = tf.constant(self.coefs_np, dtype=self.dtype_) 

    def _project(self, z):
        if self.project == "tanh":
            return tf.tanh(z)
        if self.project == "clip":
            return tf.clip_by_value(z, -1.0, 1.0)
        return z

    def call(self, inputs):
        # inputs: (Batch, Channels)
        z = tf.cast(inputs, self.dtype_)
        z = self._project(z)
        
        # ------------------------------------------------
        # 1. Bias 계산 (Channel-wise Mean)
        # ------------------------------------------------
        W = tf.nn.softmax(self.W_i, axis=-1) if self.use_softmax else self.W_i
        bias = tf.reduce_mean(W, axis=1) # (Channels,)
        
        # ------------------------------------------------
        # 2. Am (계수) 계산
        # Am shape: (Channels, Order)
        # ------------------------------------------------
        Am = tf.matmul(W, self.coefs, transpose_b=True)

        # ------------------------------------------------
        # 3. Legendre Polynomials 일괄 계산 (Vectorized Recurrence)
        # P_stack shape: (Batch, Channels, Order)
        # ------------------------------------------------
        # P_1 ~ P_order까지 미리 리스트에 저장
        # (메모리를 조금 더 쓰지만 속도는 훨씬 빠름)
        
        # 초기항
        ones = tf.ones_like(z)
        P_n_minus_2 = ones # P_0
        P_n_minus_1 = z    # P_1
        
        poly_list = []
        
        # Order 1부터 계산
        for n in range(1, self.order + 1):
            if n == 1:
                Pn = z
            else:
                # 점화식: ((2n-1)z P_{n-1} - (n-1)P_{n-2}) / n
                # (k=n-1 로 치환하여 계산)
                k = float(n - 1)
                Pn = ((2.0 * k + 1.0) * z * P_n_minus_1 - k * P_n_minus_2) / (k + 1.0)
                
                # 다음 스텝 준비
                P_n_minus_2 = P_n_minus_1
                P_n_minus_1 = Pn
            
            poly_list.append(Pn)
            
        # Stack: (Order, Batch, Channels) -> Transpose to (Batch, Channels, Order)
        P_stack = tf.stack(poly_list, axis=-1) 
        
        # ------------------------------------------------
        # 4. 최종 합산 (Einstein Summation)
        # P_stack (B, C, O) * Am (C, O) -> Sum over Order
        # ------------------------------------------------
        # term[b, c] = sum_o (P_stack[b, c, o] * Am[c, o])
        weighted_sum = tf.einsum('bco,co->bc', P_stack, Am)
        
        return weighted_sum + tf.reshape(bias, (1, -1))