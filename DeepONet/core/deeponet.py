# core/deeponet.py
import tensorflow as tf
from .lpa import LPA
from config import RE_TRAIN_LIST

def normalize_xy(xy, domain):
    # domain = (xmin, xmax, ymin, ymax)
    xmin, xmax, ymin, ymax = domain
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    x_n = 2.0 * (x - xmin) / (xmax - xmin) - 1.0
    y_n = 2.0 * (y - ymin) / (ymax - ymin) - 1.0
    return tf.concat([x_n, y_n], axis=1)

def normalize_Re(Re, Re_min, Re_max, log_scale=True):
    if log_scale:
        Re = tf.math.log(Re)
        Re_min = tf.math.log(tf.constant(Re_min, dtype=Re.dtype))
        Re_max = tf.math.log(tf.constant(Re_max, dtype=Re.dtype))
    return 2.0 * (Re - Re_min) / (Re_max - Re_min) - 1.0

def build_deeponet_backbone(
    latent_dim,
    branch_width, branch_depth,
    trunk_width, trunk_depth,
    domain,            # (xmin, xmax, ymin, ymax)
    Re_min, Re_max,    # training Re range
):
    # -----------------
    # Branch: Re
    # -----------------
    inp_branch = tf.keras.Input(shape=(1,), name="branch_input")
    b = tf.keras.layers.Lambda(
        lambda r: normalize_Re(r, Re_min, Re_max, log_scale=True),
        name="Re_norm"
    )(inp_branch)

    for _ in range(branch_depth):
        b = tf.keras.layers.Dense(branch_width, activation="tanh")(b)
    latent_b = tf.keras.layers.Dense(latent_dim, activation=None, name="branch_latent")(b)

    # -----------------
    # Trunk: (x, y)
    # -----------------
    inp_trunk = tf.keras.Input(shape=(2,), name="trunk_input")
    t = tf.keras.layers.Lambda(
        lambda xy: normalize_xy(xy, domain),
        name="xy_norm"
    )(inp_trunk)

    for _ in range(trunk_depth):
        t = tf.keras.layers.Dense(trunk_width, activation="tanh")(t)
    latent_t = tf.keras.layers.Dense(latent_dim, activation=None, name="trunk_latent")(t)

    # -----------------
    # Combine
    # -----------------
    prod = tf.keras.layers.Multiply(name="latent_prod")([latent_b, latent_t])
    return inp_branch, inp_trunk, prod

def build_model_variant_A(
    latent_dim,
    branch_width, branch_depth,
    trunk_width, trunk_depth,
    domain=(0.0, 1.0, -0.5, 1.5),
    Re_min=RE_TRAIN_LIST[0],
    Re_max=RE_TRAIN_LIST[-1],
    output_dim=3,
    use_lpa=True,
    lpa_order=3, lpa_panels=30, lpa_softmax=False,
    dtype="float32",
):
    inp_b, inp_t, prod = build_deeponet_backbone(
        latent_dim, branch_width, branch_depth,
        trunk_width, trunk_depth,
        domain, Re_min, Re_max
    )

    h = tf.keras.layers.Activation("tanh", name="prod_tanh")(prod)

    if use_lpa:
        h = LPA(
            order=lpa_order, N_p=lpa_panels,
            dtype=dtype, use_softmax=lpa_softmax,
            project="tanh",     # 권장
            name="LPA_pre_output"
        )(h)

    out = tf.keras.layers.Dense(output_dim, activation=None, name="output")(h)
    return tf.keras.Model([inp_b, inp_t], out,
        name=f"DeepONet_PINN_A_{'LPA' if use_lpa else 'VAN'}")

def build_model_variant_B(
    latent_dim,
    branch_width, branch_depth,
    trunk_width, trunk_depth,
    domain=(0.0, 1.0, -0.5, 1.5),
    Re_min=RE_TRAIN_LIST[0],
    Re_max=RE_TRAIN_LIST[-1],
    head_width=32,
    output_dim=3,
    use_lpa=True,
    lpa_order=3, lpa_panels=30, lpa_softmax=False,
    dtype="float32",
):
    inp_b, inp_t, prod = build_deeponet_backbone(
        latent_dim, branch_width, branch_depth, trunk_width, trunk_depth,
        domain, Re_min, Re_max
    )

    # [중요] 기존 PINN 구조: Hidden -> Dense -> LPA -> Output
    # 여기서 prod가 Hidden 역할, Dense가 Mixing 역할입니다.
    
    # 1. Mixing Layer (Linear)
    # Activation을 None으로 설정해야 LPA의 입력 범위가 과도하게 압축되지 않습니다.
    h = tf.keras.layers.Dense(head_width, activation=None, name="head_dense")(prod)

    if use_lpa:
        # 2. LPA (Non-linear)
        # Channel-wise LPA가 적용되어 각 채널마다 다른 다항식을 학습합니다.
        h = LPA(
            order=lpa_order, N_p=lpa_panels, dtype=dtype,
            use_softmax=lpa_softmax, project="tanh", name="LPA_pre_output"
        )(h)
    else:
        # Vanilla
        h = tf.keras.layers.Activation("tanh", name="head_tanh")(h)

    out = tf.keras.layers.Dense(output_dim, activation=None, name="output")(h)
    return tf.keras.Model([inp_b, inp_t], out, name=f"DeepONet_B_{'LPA' if use_lpa else 'VAN'}")

# core/deeponet.py 에 추가

def build_model_variant_C_TrunkLPA(
    latent_dim,
    branch_width, branch_depth,
    trunk_width, trunk_depth,
    domain=(0.0, 1.0, -0.5, 1.5),
    Re_min=RE_TRAIN_LIST[0],
    Re_max=RE_TRAIN_LIST[-1],
    output_dim=3,
    lpa_order=3, lpa_panels=10, lpa_softmax=False, # LPA 설정
    dtype="float32",
):
    # -----------------
    # 1. Branch: Re (기존과 동일하게 tanh 사용)
    # -----------------
    inp_branch = tf.keras.Input(shape=(1,), name="branch_input")
    b = tf.keras.layers.Lambda(
        lambda r: normalize_Re(r, Re_min, Re_max, log_scale=True),
        name="Re_norm"
    )(inp_branch)

    for _ in range(branch_depth):
        b = tf.keras.layers.Dense(branch_width, activation="tanh")(b)
    
    # Branch의 마지막은 Latent 차원으로 맞춤
    latent_b = tf.keras.layers.Dense(latent_dim, activation=None, name="branch_latent")(b)

    # -----------------
    # 2. Trunk: (x, y) -> LPA 적용!
    # -----------------
    inp_trunk = tf.keras.Input(shape=(2,), name="trunk_input")
    t = tf.keras.layers.Lambda(
        lambda xy: normalize_xy(xy, domain),
        name="xy_norm"
    )(inp_trunk)

    # 앞단은 섞어주는 용도로 Dense(tanh) 유지 (혹은 줄여도 됨)
    for _ in range(trunk_depth):
        t = tf.keras.layers.Dense(trunk_width, activation="tanh")(t)

    # ★ 핵심 변경: Trunk의 마지막 출력을 만들기 직전에 LPA 통과 ★
    # 먼저 Latent 차원으로 투영 (Linear)
    t = tf.keras.layers.Dense(latent_dim, activation=None, name="trunk_pre_lpa")(t)
    
    # 그 다음 LPA를 Activation처럼 사용 (Trunk의 기저를 다항식으로 변환)
    latent_t = LPA(
        order=lpa_order, N_p=lpa_panels,
        dtype=dtype, use_softmax=lpa_softmax,
        project="tanh",  # 입력 범위 안정화
        name="trunk_lpa"
    )(t)

    # -----------------
    # 3. Combine (Dot Product)
    # -----------------
    # Branch(계수) * Trunk(다항식 기저)
    prod = tf.keras.layers.Multiply(name="latent_prod")([latent_b, latent_t])

    # -----------------
    # 4. Output
    # -----------------
    # 여기서는 추가적인 비선형성 없이 바로 합쳐서 출력 (Bias 역할 Dense 하나 정도는 OK)
    out = tf.keras.layers.Dense(output_dim, activation=None, name="output")(prod)

    return tf.keras.Model([inp_branch, inp_trunk], out, name="DeepONet_TrunkLPA")