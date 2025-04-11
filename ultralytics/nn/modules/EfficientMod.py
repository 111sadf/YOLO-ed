import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple
from torch.jit import Final


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4, patch_stride=4, patch_pad=0, norm_layer=None):
        """
        In-shape [b,h,w,c], out-shape[b, h',w',c']
        Args:
            patch_size:
            in_chans:
            embed_dim:
            patch_pad:
            norm_layer:
        """
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=patch_pad)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x


class AttMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bias=True):
        # channel last
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    fast_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        # assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = max(dim // num_heads, 32)
        self.scale = self.head_dim ** -0.5
        self.fast_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  # FIXME

        self.qkv = nn.Linear(dim, self.num_heads * self.head_dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fast_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.head_dim * self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class AttentionBlock(nn.Module):

    def __init__(
            self,
            dim, mlp_ratio=4., num_heads=8, qkv_bias=False, qk_norm=False, drop=0., attn_drop=0.,
            init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = AttMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, H, W, C = x.size()
        x = x.reshape(B, H * W, C).contiguous()
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        x = x.reshape(B, H, W, C).contiguous()
        return x


class ContextLayer(nn.Module):
    def __init__(self, in_dim, conv_dim, context_size=[3], context_act=nn.GELU,
                 context_f=True, context_g=True):
        # channel last
        super().__init__()
        self.f = nn.Linear(in_dim, conv_dim) if context_f else nn.Identity()
        self.g = nn.Linear(conv_dim, in_dim) if context_g else nn.Identity()
        self.context_size = context_size
        self.act = context_act() if context_act else nn.Identity()
        if not isinstance(context_size, (list, tuple)):
            context_size = [context_size]
        self.context_list = nn.ModuleList()
        for c_size in context_size:
            self.context_list.append(
                nn.Conv2d(conv_dim, conv_dim, c_size, stride=1, padding=c_size // 2, groups=conv_dim)
            )

    def forward(self, x):
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        out = 0
        for i in range(len(self.context_list)):
            ctx = self.act(self.context_list[i](x))
            out = out + ctx
        out = self.g(out.permute(0, 2, 3, 1).contiguous())
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., bias=True, conv_in_mlp=True,
                 conv_group_dim=4, context_size=3, context_act=nn.GELU,
                 context_f=True, context_g=True):
        # channel last
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.conv_in_mlp = conv_in_mlp
        if self.conv_in_mlp:
            self.conv_group_dim = conv_group_dim
            self.conv_dim = hidden_features // conv_group_dim
            self.context_layer = ContextLayer(in_features, self.conv_dim, context_size,
                                              context_act, context_f, context_g)

        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

        if hidden_features == in_features and conv_group_dim == 1:
            self.expand_dim = False
        else:
            self.expand_dim = True
            self.act = act_layer()
            self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.conv_in_mlp:
            conv_x = self.context_layer(x)
        x = self.fc1(x)
        if self.expand_dim:
            x = self.act(x)
            x = self.drop(x)
        if self.conv_in_mlp:
            if self.expand_dim:
                x = x * conv_x.repeat(1, 1, 1, self.conv_group_dim)
            else:
                x = x * conv_x
        x = self.fc2(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., conv_in_mlp=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 bias=True, use_layerscale=False, layerscale_value=1e-4,
                 conv_group_dim=4, context_size=3, context_act=nn.GELU,
                 context_f=True, context_g=True
                 ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop, bias=bias, conv_in_mlp=conv_in_mlp,
                       conv_group_dim=conv_group_dim, context_size=context_size, context_act=context_act,
                       context_f=context_f, context_g=context_g)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_1 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        shortcut = x
        x = shortcut + self.drop_path(self.gamma_1 * self.mlp(self.norm(x)))
        return x