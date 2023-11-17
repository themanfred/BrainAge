import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.layers import Dense, Dropout, Conv2D, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.schedules import ExponentialDecay
#from tensorflow.keras.losses import huber

# Set GPU device configuration
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

CFGS = {
    'swin_tiny_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]),
    'swin_small_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
    'swin_base_224': dict(input_size=(224, 224), window_size=7, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_base_384': dict(input_size=(384, 384), window_size=12, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_large_224': dict(input_size=(224, 224), window_size=7, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48]),
    'swin_large_384': dict(input_size=(384, 384), window_size=12, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48])
}


def huber_loss(y_true, y_pred, delta=1):
    error = y_pred - y_true
    abs_error = K.abs(error)
    quadratic = K.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * K.square(quadratic) + delta * linear



class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., prefix=''):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Dense(hidden_features, name=f'{prefix}/mlp/fc1')
        self.fc2 = Dense(out_features, name=f'{prefix}/mlp/fc2')
        self.drop = Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = tf.keras.activations.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.get_shape().as_list() #retrives  the dimensions of the input tensor x
     #The tensor is reshaped to group the elements that will form the windows. 
    #It is divided along the height and width by window_size to create a new dimension that will represent the windows. 
    #This results in a 6D tensor where the third and fifth dimensions are the height and width of the window, respectively.
    x = tf.reshape(x, shape=[-1, H // window_size,
                   window_size, W // window_size, window_size, C]) 
   
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=[-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    x = tf.reshape(windows, shape=[-1, H // window_size,
                   W // window_size, window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=[-1, H, W, C])
    return x


class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., prefix=''):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.prefix = prefix

        self.qkv = Dense(dim * 3, use_bias=qkv_bias,
                         name=f'{self.prefix}/attn/qkv')
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name=f'{self.prefix}/attn/proj')
        self.proj_drop = Dropout(proj_drop)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(f'{self.prefix}/attn/relative_position_bias_table',
                                                            shape=(
                                                                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads),
                                                            initializer=tf.initializers.Zeros(), trainable=True)

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int64)
        self.relative_position_index = tf.Variable(initial_value=tf.convert_to_tensor(
            relative_position_index), trainable=False, name=f'{self.prefix}/attn/relative_position_index')
        self.built = True

    def call(self, x, mask=None):
        B, N, C = x.get_shape().as_list()
        qkv = tf.transpose(tf.reshape(self.qkv(
            x), shape=[-1, N, 3, self.num_heads, C // self.num_heads]), perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))
        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(
            self.relative_position_index, shape=[-1]))
        relative_position_bias = tf.reshape(relative_position_bias, shape=[
                                            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])
        relative_position_bias = tf.transpose(
            relative_position_bias, perm=[2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]  # tf.shape(mask)[0]
            attn = tf.reshape(attn, shape=[-1, nW, self.num_heads, N, N]) + tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), attn.dtype)
            attn = tf.reshape(attn, shape=[-1, self.num_heads, N, N])
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)

        attn = self.attn_drop(attn)

        x = tf.transpose((attn @ v), perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs

    # Compute keep_prob
    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    random_tensor = keep_prob
    shape = (tf.shape(inputs)[0],) + (1,) * \
        (len(tf.shape(inputs)) - 1)
    random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0., norm_layer=LayerNormalization, prefix=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.prefix = prefix

        self.norm1 = norm_layer(epsilon=1e-5, name=f'{self.prefix}/norm1')
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, prefix=self.prefix)
        self.drop_path = DropPath(
            drop_path_prob if drop_path_prob > 0. else 0.)
        self.norm2 = norm_layer(epsilon=1e-5, name=f'{self.prefix}/norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       drop=drop, prefix=self.prefix)

    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = np.zeros([1, H, W, 1])
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            img_mask = tf.convert_to_tensor(img_mask)
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(
                mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask, trainable=False, name=f'{self.prefix}/attn_mask')
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=[-1, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=[-1, self.window_size * self.window_size, C])

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = tf.reshape(
            attn_windows, shape=[-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[
                        self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x
        x = tf.reshape(x, shape=[-1, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, input_resolution, dim, norm_layer=LayerNormalization, prefix=''):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = Dense(2 * dim, use_bias=False,
                               name=f'{prefix}/downsample/reduction')
        self.norm = norm_layer(epsilon=1e-5, name=f'{prefix}/downsample/norm')

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = tf.reshape(x, shape=[-1, H, W, C])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(x, shape=[-1, (H // 2) * (W // 2), 4 * C])

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_prob=0., norm_layer=LayerNormalization, downsample=None, use_checkpoint=False, prefix=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = tf.keras.Sequential([SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                           num_heads=num_heads, window_size=window_size,
                                           shift_size=0 if (
                                               i % 2 == 0) else window_size // 2,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path_prob=drop_path_prob[i] if isinstance(
                                               drop_path_prob, list) else drop_path_prob,
                                           norm_layer=norm_layer,
                                           prefix=f'{prefix}/blocks{i}') for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer, prefix=prefix)
        else:
            self.downsample = None

    def call(self, x):
        x = self.blocks(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x

#changed in_chan from 3 to 1
class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, img_size=(224, 224), patch_size=(4, 4), in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__(name='patch_embed')
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv2D(embed_dim, kernel_size=patch_size,
                           strides=patch_size, name='proj')
        if norm_layer is not None:
            self.norm = norm_layer(epsilon=1e-5, name='norm')
        else:
            self.norm = None

    def call(self, x):
        print('--', x.shape, x.dtype)
        B, H, W, C = x.get_shape().as_list()
        # Print or use tf.print to inspect the input size
        print(f"Input image shape: ({B}, {H}, {W}, {C})")
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)
        x = tf.reshape(
            x, shape=[-1, (H // self.patch_size[0]) * (W // self.patch_size[0]), self.embed_dim])
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerModel(tf.keras.Model):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', include_top=False,
                 img_size=(224, 224), patch_size=(4, 4), in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=LayerNormalization, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__(name=model_name)

        self.include_top = include_top

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute postion embedding
        if self.ape:
            self.absolute_pos_embed = self.add_weight('absolute_pos_embed',
                                                      shape=(
                                                          1, num_patches, embed_dim),
                                                      initializer=tf.initializers.Zeros())

        self.pos_drop = Dropout(drop_rate)

        # stochastic depth
        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

        # build layers
        self.basic_layers = tf.keras.Sequential([BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                  patches_resolution[1] // (2 ** i_layer)),
                                                depth=depths[i_layer],
                                                num_heads=num_heads[i_layer],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path_prob=dpr[sum(depths[:i_layer]):sum(
                                                    depths[:i_layer + 1])],
                                                norm_layer=norm_layer,
                                                downsample=PatchMerging if (
                                                    i_layer < self.num_layers - 1) else None,
                                                use_checkpoint=use_checkpoint,
                                                prefix=f'layers{i_layer}') for i_layer in range(self.num_layers)])
        self.norm = norm_layer(epsilon=1e-5, name='norm')
        self.avgpool = GlobalAveragePooling1D()
        if self.include_top:
            self.head = Dense(num_classes, name='head')
        else:
            self.head = None

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x = self.basic_layers(x)
        x = self.norm(x)
        x = self.avgpool(x)
        return x

    def call(self, x):
        x = self.forward_features(x)
        if self.include_top:
            x = self.head(x)
        return x

# removed num_classes=1000 since it is not relevant
def SwinTransformer(model_name='swin_tiny_224', num_classes=1000 ,include_top=True, pretrained=False, use_tpu=False, cfgs=CFGS, in_chans=1):
    cfg = cfgs[model_name]
    net = SwinTransformerModel(
        model_name=model_name, include_top=include_top, num_classes=num_classes, img_size=cfg['input_size'], window_size=cfg[
            'window_size'], embed_dim=cfg['embed_dim'], depths=cfg['depths'], num_heads=cfg['num_heads']
    )
    net(tf.keras.Input(shape=(cfg['input_size'][0], cfg['input_size'][1], in_chans)))
    if pretrained is True:
        url = f'https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/{model_name}.tgz'
        pretrained_ckpt = tf.keras.utils.get_file(
            model_name, url, untar=True)
    else:
        pretrained_ckpt = pretrained

    if pretrained_ckpt:
        if tf.io.gfile.isdir(pretrained_ckpt):
            pretrained_ckpt = f'{pretrained_ckpt}/{model_name}.ckpt'

        if use_tpu:
            load_locally = tf.saved_model.LoadOptions(
                experimental_io_device='/job:localhost')
            net.load_weights(pretrained_ckpt, options=load_locally)
        else:
            net.load_weights(pretrained_ckpt)

    return net

def build_ST():
    IN_CHANNELS = 1
    NUM_CLASSES = 1
    IMAGE_SIZE = [224, 224]
    
    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda data: tf.cast(data, tf.float32), input_shape=[*IMAGE_SIZE, IN_CHANNELS]),
        tf.keras.layers.Lambda(lambda x: x / 255.0),  # Normalize the pixel value
        SwinTransformer(model_name='swin_tiny_224', include_top=False, pretrained=False, in_chans=IN_CHANNELS),
        tf.keras.layers.Dense(
            NUM_CLASSES, 
            activation='linear', 
            kernel_initializer=tf.keras.initializers.HeNormal()  # He normal initializer for weights
        )
    ])

    # Add a print statement to inspect the model
    print("Model Summary:")
    model.summary()

    return model


"""
def build_model_test():
    import os
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
    model = build_ST()
    #changed to a more updated tf and keras were decay has been replaced with learning_rate_decay and the previous value of deccay was .003 while learning rate was the same... So I changed that to 1E-6
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, learning_rate_decay=1e-6), loss=huber, metrics=['mae'])
    random_tensor = tf.random.normal((4, 224, 224, 1))
    outputs = model(random_tensor, training=True)
    model.summary()
    print(outputs)
    
    return model

"""
#modified version
def build_model_test():
    model = build_ST()
    
    # Define the learning rate schedule using ExponentialDecay
    initial_learning_rate = 1e-3
    decay_steps = 10000  # You can adjust this value as needed
    decay_rate = 0.9  # You can adjust this value as needed
    learning_rate_schedule = ExponentialDecay(initial_learning_rate, decay_steps, decay_rate)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    
    model.compile(optimizer=optimizer, loss=huber_loss, metrics=['mae'])
    
    random_tensor = tf.random.normal((4, 224, 224, 1))
    outputs = model(random_tensor, training=True)
    model.summary()
    print(outputs)
    
    return model
"""

class ConvolutionalArea(tf.keras.layers.Layer):
    def __init__(self, initial_filters=1, **kwargs):
        super(ConvolutionalArea, self).__init__(**kwargs)
        self.initial_filters = initial_filters

    def call(self, inputs):
        x = inputs
        n = self.initial_filters

        # First group of layers
        x = tf.keras.layers.Conv2D(n, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(n, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
        # Second group of layers
        x = tf.keras.layers.Conv2D(n*2, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(n*2, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
        # Third group of layers
        x = tf.keras.layers.Conv2D(n*4, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(n*4, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        return x

class ABCConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size, **kwargs):
        super(ABCConvolutionLayer, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # Define the convolutional layer here.
        self.base_conv = Conv2D(self.out_channels, self.kernel_size, padding='same')

    def call(self, inputs):
        # Apply the convolution operation.
        x = self.base_conv(inputs)
        return x




class ConvolutionalArea(tf.keras.layers.Layer):
    def __init__(self, initial_filters=2, **kwargs):
        super(ConvolutionalArea, self).__init__(**kwargs)
        self.initial_filters = initial_filters

  def call(self, inputs):
        x = inputs
        n = self.initial_filters

        # Convolutional Area
        for i in range(8):
            x = tf.keras.layers.Conv2D(n, (3, 3), activation='relu', padding='same')(x)
            
            # Increase filters for every second convolution
            if (i+1) % 2 == 0 and i < 6:  # Adjusted logic to ensure the last convolution doesn't double the filters
                n *= 2
            
            # Max-pooling layers after the 2nd and 4th convolutional layers
            if i == 1 or i == 3:
                x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
                
        return x



class ShiftedSelfAttentionArea(tf.keras.layers.Layer):
    def __init__(self, cQ, cK, cV, window_size=7, **kwargs):
        super(ShiftedSelfAttentionArea, self).__init__(**kwargs)
        self.cQ = cQ
        self.cK = cK
        self.cV = cV
        self.window_size = window_size

    def build(self, input_shape):
        self.batch_size, self.height, self.width, self.channels = input_shape
        self.Q_layer = Conv2D(n=self.cQ, kernel_size=(1, 1), padding='same')
        self.K_layer = Conv2D(n=self.cK, kernel_size=(1, 1), padding='same')
        self.V_layer = Conv2D(n=self.cV, kernel_size=(1, 1), padding='same')

    def call(self, inputs):
        Q = self.Q_layer(inputs)
        K = self.K_layer(inputs)
        V = self.V_layer(inputs)

        # Flatten tensors to perform the dot product
        Q_flat = Reshape((self.height * self.width, self.cQ))(Q)
        K_flat = Reshape((self.height * self.width, self.cK))(K)
        V_flat = Reshape((self.height * self.width, self.cV))(V)
        
        # Computing attention scores and weights
        attention_scores = tf.matmul(Q_flat, K_flat, transpose_b=True) / tf.math.sqrt(float(self.cK))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_weights, V_flat)
        
        # Reshaping the output back to its 2D form
        attention_output_reshaped = Reshape((self.height, self.width, self.cV))(attention_output)

        # Shifted window partitioning
        shift = self.window_size // 2
        shifted_inputs = inputs[:, shift:-shift, shift:-shift, :]
        
        # Apply self attention on the shifted input
        attention_output_shifted = attention_output_reshaped[:, shift:-shift, shift:-shift, :]

        # Combine the shifted attention output with the original windowed feature map
        attention_output = inputs - attention_output_shifted

        return attention_output




# Define the convolutional area
def convolutional_area(input_tensor):
   x = input_tensor
    for i in range(8):
        if i % 2 == 0:  # Using ABCConvolution for every alternate layer as an example
            x = ABCConvolutionLayer(64, (3,3))(x)
        else:
            x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
        
        if i == 1 or i == 3:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    return x


def self_attention_area(x, cQ, cK, cV):
    assert cV == CHANNELS_AFTER_SELF_ATTENTION, "cV should match CHANNELS_AFTER_SELF_ATTENTION"
    
    batch_size, height, width, channels = x.shape
    flattened_x = tf.reshape(x, (batch_size, height * width, channels))
    
    # Use cQ, cK, and cV for the respective Conv1D layers
    Q = tf.keras.layers.Conv1D(n=cQ, kernel_size=1)(flattened_x)
    K = tf.keras.layers.Conv1D(n=cK, kernel_size=1)(flattened_x)
    V = tf.keras.layers.Conv1D(n=cV, kernel_size=1)(flattened_x)
    
    attention_scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(float(cK))
    attention_weights = tf.nn.softmax(attention_scores, axis=-1)
    
    attention_output = tf.matmul(attention_weights, V)
    attention_output_reshaped = tf.reshape(attention_output, (batch_size, height, width, cV))
    
    return attention_output_reshaped


def build_ST():
    
    IN_CHANNELS = 1
    NUM_CLASSES = 1
    IMAGE_SIZE = [224, 224]

    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda data: tf.cast(data, tf.float32), input_shape=[*IMAGE_SIZE, IN_CHANNELS]),
        tf.keras.layers.Lambda(lambda x: x / 255.0), #normalize the pixel value
        SwinTransformer(model_name='swin_tiny_224', include_top=False, pretrained=False, in_chans=IN_CHANNELS),
        tf.keras.layers.Dense(NUM_CLASSES, activation='linear')
    ])
    
    return model

# Test the functional model architecture
model_functional = build_ST()
model_functional.summary()

     

    CHANNELS_AFTER_SELF_ATTENTION = 16

    # Define the input tensor
    input_tensor = tf.keras.Input(shape=(*IMAGE_SIZE, IN_CHANNELS))
   
    # Convolutional Area
    conv_output = ConvolutionalArea()(input_tensor)
    
    # Self Attention Area (ABC)
    attention_output_reshaped = SelfAttentionArea(cQ=64, cK=64, cV=CHANNELS_AFTER_SELF_ATTENTION)(conv_output)
    
    # Patch Embedding for Swin Transformer
    patch_embedded_output = PatchEmbed(img_size=IMAGE_SIZE, patch_size=(4, 4), in_chans=IN_CHANNELS, embed_dim=96)(input_tensor)
  
    # Reshape the attention_output_reshaped tensor
    attention_output_reshaped = Reshape((-1, 3136, 16))(attention_output_reshaped)
    attention_output_reshaped = tf.squeeze(attention_output_reshaped, axis=1)  # Squeeze out the extra dimension    

    # Concatenate the reshaped tensor with patch_embedded_output
    concatenated_input = tf.concat([attention_output_reshaped, patch_embedded_output], axis=-1)
    
    # Swin Transformer
    transformer_output = SwinTransformer(model_name='swin_tiny_224', include_top=False, pretrained=False, in_chans=IN_CHANNELS)(concatenated_input)
    
    # Dense Layer
    output = tf.keras.layers.Dense(NUM_CLASSES, activation='linear')(transformer_output)
    
    # Build the model
    model = tf.keras.Model(inputs=input_tensor, outputs=output)
    """



"""
def build_model_test():
    import os
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
    model = build_ST()
    #changed to a more updated tf and keras were decay has been replaced with learning_rate_decay and the previous value of deccay was .003 while learning rate was the same... So I changed that to 1E-6
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, learning_rate_decay=1e-6), loss=huber, metrics=['mae'])
    random_tensor = tf.random.normal((4, 224, 224, 1))
    outputs = model(random_tensor, training=True)
    model.summary()
    print(outputs)
    
    return model

def test_abc_convolution():
    # Initialize the ABCConvolution layer
    abc_conv = ABCConvolution(64, 3, stride=1, padding='same')
    
    # Generate random input tensor
    input_tensor = tf.random.normal([4,224,224, 1])  # Batch of 8 images with 32x32 size and 3 channels
    
    # Apply ABCConvolution
    output, attentions = abc_conv(input_tensor)
    
    return output.shape, attentions.shape

# Run the test
output_shape, attention_shape = test_abc_convolution()
print("Output Shape:", output_shape)
print("Attention Shape:", attention_shape)


#modified version
def build_model_test():
    model = build_ST()
    
    # Define the learning rate schedule using ExponentialDecay
    initial_learning_rate = 1e-3
    decay_steps = 10000  # You can adjust this value as needed
    decay_rate = 0.9  # You can adjust this value as needed
    learning_rate_schedule = ExponentialDecay(initial_learning_rate, decay_steps, decay_rate)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    
    model.compile(optimizer=optimizer, loss=huber_loss, metrics=['mae'])
    
    random_tensor = tf.random.normal((4, 224, 224, 1))
    outputs = model(random_tensor, training=True)
    model.summary()
    print(outputs)
    
    return model
""" 
