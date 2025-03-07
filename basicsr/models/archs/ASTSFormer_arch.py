import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import math
from einops import rearrange
from .pair_wise_distance import PairwiseDistFunction
from skimage.segmentation import slic,mark_boundaries,find_boundaries
from skimage.measure import regionprops
import numpy as np
from skimage.morphology import label
from skimage.measure import block_reduce
from skimage.transform import resize


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)




# Mixed-scale Gated Feed-forward Network (MGFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        # Project input to hidden features * 2
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # Depthwise convolution for x1 branch
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)

        # Linear layer for x2 branch
        self.linear = nn.Linear(hidden_features, hidden_features, bias=bias)

        # Convolution to adjust channel dimension after concat
        self.adjust_conv = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=1, bias=bias)

        # Project output back to original dimension
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # Project input to hidden features * 2
        x = self.project_in(x)

        # Split into two branches
        x1, x2 = x.chunk(2, dim=1)

        # Process x1 branch with depthwise convolution
        x1 = self.dwconv(x1)

        # Process x2 branch with linear transformation
        batch_size, channels, height, width = x2.shape
        x2 = x2.permute(0, 2, 3, 1)  # [batch_size, height, width, channels]
        x2 = self.linear(x2)  # Apply linear transformation
        x2 = x2.permute(0, 3, 1, 2)  # [batch_size, channels, height, width]

        # Concat x1 and x2
        x1 = torch.cat([x1, x2], dim=1)  # [batch_size, 2 * channels, height, width]
        x2 = torch.cat([x1, x2], dim=1)  # [batch_size, 3 * channels, height, width]

        # Adjust channel dimension to make x1 and x2 compatible
        x1 = self.adjust_conv(x1)  # [batch_size, channels, height, width]
        x2 = self.adjust_conv(x2)  # [batch_size, channels, height, width]

        # Combine branches with GELU activation
        x = F.gelu(x1) * x2

        # Project output back to original dimension
        x = self.project_out(x)

        return x


# Sparse-token Selective Attention (SSSA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.attn_drop = nn.Dropout(0.)
        # [x2, x3, x4] -> [96, 72, 48]
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.kernel_size = [48, 48]
        self.proj = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.w = nn.Parameter(torch.ones(2)) 
    def forward(self, x, superpixel_features):
        b, c, h, w = x.shape
        #print(superpixel_features.device) 
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        B, C, H, W = q.shape
        q1 = torch.nn.functional.normalize(q, dim=-1)
        superpixel_features_norm = torch.nn.functional.normalize(superpixel_features,dim=-1)
        superpixel_features_expanded = superpixel_features_norm.repeat(B, 1, 1, 1)
        superpixel_features_mean = self.proj(superpixel_features_expanded)
        q_similarity_scores = (q1 * superpixel_features_mean)
        q_similarity_scores = rearrange(q_similarity_scores, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_similarity_scores = torch.nn.functional.normalize(q_similarity_scores, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q_similarity_scores.shape  

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q_similarity_scores @ k.transpose(-2, -1)) * self.temperature  # b 1 C C


        small_constant = 1e-6
        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, attn * small_constant)

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, attn * small_constant)

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, attn * small_constant)

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, attn * small_constant)

        attn1 = attn1.softmax(dim=-1)  # [1 6 30 30]
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        attn0 = self.relu(attn)

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        
        attn1 = attn0*w1+attn1*w2
        attn1 = self.attn_drop(attn1)
        
        attn2 = attn0*w1+attn2*w2
        attn2 = self.attn_drop(attn2)
    
        attn3 = attn0*w1+attn3*w2
        attn3 = self.attn_drop(attn3)
        
        attn4 = attn0*w1+attn4*w2
        attn4 = self.attn_drop(attn4)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=qkv.shape[-2], w=qkv.shape[-1])

        out = self.project_out(out)
        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, superpixel_features):
        x = x + self.attn(self.norm1(x),superpixel_features)
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Downsample_sp(nn.Module):
    def __init__(self, n_feat):
        super(Downsample_sp, self).__init__()
        
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        x = x.unsqueeze(0)
        #print('x.shape:',x.shape)
        return self.body(x)

class Downsample_sp_1(nn.Module):
    def __init__(self, n_feat):
        super(Downsample_sp_1, self).__init__()
        
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample_sp(nn.Module):
    def __init__(self, n_feat):
        super(Upsample_sp, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        x = x.unsqueeze(0)
        return self.body(x)

class Upsample_sp_1(nn.Module):
    def __init__(self, n_feat):
        super(Upsample_sp_1, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
#########################################
########### SSM ################
@torch.no_grad()
def get_abs_indices(init_label_map, num_spixels_width):
    b, n_pixel = init_label_map.shape
    device = init_label_map.device
    r = torch.arange(-1, 2.0, device=device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)

    abs_pix_indices = torch.arange(n_pixel, device=device)[None, None].repeat(b, 9, 1).reshape(-1).long()
    abs_spix_indices = (init_label_map[:, None] + relative_spix_indices[None, :, None]).reshape(-1).long()
    abs_batch_indices = torch.arange(b, device=device)[:, None, None].repeat(1, 9, n_pixel).reshape(-1).long()

    return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)
@torch.no_grad()
def get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width):
    relative_label = affinity_matrix.max(1)[1]
    r = torch.arange(-1, 2.0, device=affinity_matrix.device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)
    label = init_label_map + relative_spix_indices[relative_label]
    return label.long()
def calc_init_centroid(images, num_spixels_width, num_spixels_height):
    """
    calculate initial superpixels
    Args:
        images: torch.Tensor
            A Tensor of shape (B, C, H, W)
        spixels_width: int
            initial superpixel width
        spixels_height: int
            initial superpixel height
    Return:
        centroids: torch.Tensor
            A Tensor of shape (B, C, H * W)
        init_label_map: torch.Tensor
            A Tensor of shape (B, H * W)
        num_spixels_width: int
            A number of superpixels in each column
        num_spixels_height: int
            A number of superpixels int each raw
    """
    batchsize, channels, height, width = images.shape
    device = images.device

    centroids = torch.nn.functional.adaptive_avg_pool2d(images, (num_spixels_height, num_spixels_width))

    with torch.no_grad():
        num_spixels = num_spixels_width * num_spixels_height
        labels = torch.arange(num_spixels, device=device).reshape(1, 1, *centroids.shape[-2:]).type_as(centroids)
        init_label_map = torch.nn.functional.interpolate(labels, size=(height, width), mode="nearest")
        init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)

    init_label_map = init_label_map.reshape(batchsize, -1)
    centroids = centroids.reshape(batchsize, channels, -1)

    return centroids, init_label_map
def ssn_iter(pixel_features, stoken_size=[16, 16], n_iter=2):
    #print('pixel_features0.shape:',pixel_features.shape)
    """
    computing assignment iterations
    detailed process is in Algorithm 1, line 2 - 6
    Args:
        pixel_features: torch.Tensor
            A Tensor of shape (B, C, H, W)
        num_spixels: int
            A number of superpixels
        n_iter: int
            A number of iterations
        return_hard_label: bool
            return hard assignment or not
    """
    height, width = pixel_features.shape[-2:]
    sheight, swidth = stoken_size
    num_spixels_height = height // sheight
    num_spixels_width = width // swidth
    num_spixels = num_spixels_height * num_spixels_width

    # import pdb; pdb.set_trace()
    # num_spixels_width = int(math.sqrt(num_spixels * width / height))
    # num_spixels_height = int(math.sqrt(num_spixels * height / width))

    spixel_features, init_label_map = \
        calc_init_centroid(pixel_features, num_spixels_width, num_spixels_height)
    abs_indices = get_abs_indices(init_label_map, num_spixels_width)

    pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)
    #print('pixel_features.shape1:',pixel_features.shape)
    permuted_pixel_features = pixel_features.permute(0, 2, 1).contiguous()

    with torch.no_grad():
        for k in range(n_iter):
            if k < n_iter - 1:

                dist_matrix = PairwiseDistFunction.apply(
                    pixel_features, spixel_features, init_label_map, num_spixels_width, num_spixels_height)

                affinity_matrix = (-dist_matrix).softmax(1)
                reshaped_affinity_matrix = affinity_matrix.reshape(-1)

                mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)
                sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask])

                abs_affinity = sparse_abs_affinity.to_dense().contiguous()
                spixel_features = torch.bmm(abs_affinity, permuted_pixel_features) \
                                  / (abs_affinity.sum(2, keepdim=True) + 1e-16)

                spixel_features = spixel_features.permute(0, 2, 1).contiguous()
            else:
                dist_matrix = PairwiseDistFunction.apply(
                    pixel_features, spixel_features, init_label_map, num_spixels_width, num_spixels_height)

                affinity_matrix = (-dist_matrix).softmax(1)
                reshaped_affinity_matrix = affinity_matrix.reshape(-1)

                mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)
                sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask])

                abs_affinity = sparse_abs_affinity.to_dense().contiguous()
    def visualize_spixels(image, soft_association, num_spixels, stoken_size):
        _, h, w = image.shape
        sheight, swidth = stoken_size
        spixel_image = torch.zeros_like(image)
        soft_labels = soft_association.argmax(1).view(-1, h, w)
        for i in range(num_spixels):
            mask = soft_labels == i
            selected_mask = mask[0]
            for channel in range(spixel_image.shape[0]):
                mean_val = image[channel,selected_mask].mean()
                spixel_image[channel,selected_mask] = mean_val
        return spixel_image
    soft_association, num_spixels = abs_affinity,num_spixels
    stoken_size = [16, 16]
    n = pixel_features.shape[2]
    h = height
    w = width
    pixel_features = rearrange(pixel_features,'b c (h w) -> b c h w',h=h,w=w)
    image = pixel_features.squeeze()
    def merge_images(images):
        return images[0]
    tensor_dim = image.dim()
    if tensor_dim == 4:
        image = merge_images([image[i] for i in range(image.shape[0])])
    if image.dim() < 3:
        image = image.unsqueeze(0) 
    spixel_image = visualize_spixels(image, soft_association, num_spixels, stoken_size)
    spixel_image = spixel_image.unsqueeze(0)
    return spixel_image

# GenSP
class GenSP(nn.Module):
    def __init__(self, n_iter=3):
        super().__init__()

        self.n_iter = n_iter

    def forward(self, x, stoken_size):
        segments_slic = ssn_iter(x, stoken_size, self.n_iter)

        return segments_slic


class BasicASTSFormerLayer(nn.Module):
    def __init__(self, dim,num_heads,ffn_expansion_factor,bias,LayerNorm_type,depth):
        super(BasicASTSFormerLayer, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim,
                                num_heads=num_heads,
                                ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type
                                )
            for i in range(depth)])
    
    def forward(self, x, segments):
        for block in self.blocks:
            x = block(x, segments)  
        return x

##########################################################################
##---------- ASTSFormer -----------------------
class ASTSFormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        stoken_size=[16,16],
        dim = 48,
        dim1 = 3,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(ASTSFormer, self).__init__()
        self.stoken_size = stoken_size
        self.gen_super_pixel = GenSP(3)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = BasicASTSFormerLayer(dim=dim, 
                                                  num_heads=heads[0], 
                                                  ffn_expansion_factor=ffn_expansion_factor, 
                                                  bias=bias, 
                                                  LayerNorm_type=LayerNorm_type, 
                                                  depth=num_blocks[0])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.down1_2_1 = Downsample_sp(dim1) ## From Level 1 to Level 2
        self.encoder_level2 = BasicASTSFormerLayer(dim=int(dim*2**1), 
                                                  num_heads=heads[1], 
                                                  ffn_expansion_factor=ffn_expansion_factor, 
                                                  bias=bias, LayerNorm_type=LayerNorm_type,
                                                  depth=num_blocks[1])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.down2_3_1 = Downsample_sp_1(4) ## From Level 2 to Level 3
        self.encoder_level3 = BasicASTSFormerLayer(dim=int(dim*2**2), 
                                                  num_heads=heads[2], 
                                                  ffn_expansion_factor=ffn_expansion_factor, 
                                                  bias=bias, 
                                                  LayerNorm_type=LayerNorm_type,
                                                  depth=num_blocks[2])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.down3_4_1 = Downsample_sp_1(8) ## From Level 3 to Level 4
        self.latent = BasicASTSFormerLayer(dim=int(dim*2**3), 
                                          num_heads=heads[3], 
                                          ffn_expansion_factor=ffn_expansion_factor, 
                                          bias=bias, 
                                          LayerNorm_type=LayerNorm_type,
                                          depth=num_blocks[3])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.up4_3_1 = Upsample_sp_1(16) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = BasicASTSFormerLayer(dim=int(dim*2**2), 
                                                  num_heads=heads[2], 
                                                  ffn_expansion_factor=ffn_expansion_factor, 
                                                  bias=bias, LayerNorm_type=LayerNorm_type,
                                                  depth=num_blocks[2])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.up3_2_1 = Upsample_sp_1(8) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = BasicASTSFormerLayer(dim=int(dim*2**1), 
                                                  num_heads=heads[1], 
                                                  ffn_expansion_factor=ffn_expansion_factor, 
                                                  bias=bias, 
                                                  LayerNorm_type=LayerNorm_type,
                                                  depth=num_blocks[1])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_1_1 = Upsample_sp_1(4)  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = BasicASTSFormerLayer(dim=int(dim*2**1), 
                                                  num_heads=heads[0], 
                                                  ffn_expansion_factor=ffn_expansion_factor, 
                                                  bias=bias, 
                                                  LayerNorm_type=LayerNorm_type,
                                                  depth=num_blocks[0])
        
        self.refinement = BasicASTSFormerLayer(dim=int(dim*2**1), 
                                              num_heads=heads[0], 
                                              ffn_expansion_factor=ffn_expansion_factor, 
                                              bias=bias, LayerNorm_type=LayerNorm_type,
                                              depth=num_refinement_blocks)
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        
    def forward(self, inp_img):
        segments_slic = self.gen_super_pixel(inp_img,self.stoken_size)
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1,segments_slic)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        segments_slic = pool1(segments_slic)
        out_enc_level2 = self.encoder_level2(inp_enc_level2,segments_slic)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        segments_slic = pool2(segments_slic)
        out_enc_level3 = self.encoder_level3(inp_enc_level3,segments_slic) 
        inp_enc_level4 = self.down3_4(out_enc_level3)
        pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        segments_slic = pool3(segments_slic)      
        latent = self.latent(inp_enc_level4,segments_slic)          
        inp_dec_level3 = self.up4_3(latent)

        segments_slic = F.interpolate(segments_slic, scale_factor=2,mode='bilinear',align_corners=False)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3,segments_slic) 
        inp_dec_level2 = self.up3_2(out_dec_level3)
        segments_slic = F.interpolate(segments_slic, scale_factor=2,mode='bilinear',align_corners=False)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2,segments_slic) 
        inp_dec_level1 = self.up2_1(out_dec_level2)

        segments_slic = F.interpolate(segments_slic, scale_factor=2,mode='bilinear',align_corners=False)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1,segments_slic)
        out_dec_level1 = self.refinement(out_dec_level1,segments_slic)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img
        
        return out_dec_level1