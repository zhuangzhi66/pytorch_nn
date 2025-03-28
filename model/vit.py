from functools import partial
from typing import OrderedDict
import torch
import torch.nn as nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:  # drop_prob废弃率=0，或者不是训练的时候，就保持原来不变
        return x
    keep_prob = 1 - drop_prob  # 保持率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (b, 1, 1, 1) 元组  ndim 表示几维，图像为4维
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # 0-1之间的均匀分布[2,1,1,1]
    random_tensor.floor_()  # 下取整从而确定保存哪些样本 总共有batch个数
    output = x.div(keep_prob) * random_tensor  # 除以 keep_prob 是为了让训练和测试时的期望保持一致
    # 如果keep，则特征值除以 keep_prob；如果drop，则特征值为0
    return output  # 与x的shape保持不变

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.model):
    def __init__(self,img_size=224,patch_size=16,in_c=3,embed_dim=768,norm_layer=1):
        super().__init__()
        img_size=(img_size,img_size)    #输入图像大小变为二维元组
        patch_size=(patch_size,patch_size)
        self.img_size=img_size
        self.patch_size=patch_size
        #计算patch的网格大小,224/16=14
        self.grid_size=(img_size[0]//patch_size[0],img_size[1]//patch_size[1])
        self.num_patch=self.grid_size[0]*self.grid_size[1]

        #B,3,224,224->B,768,14,14
        self.proj=nn.Conv2d(in_c,embed_dim,kernel_size=patch_size,stride=patch_size)
        self.norm=norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self,x):
        B,C,H,W=x.shape #获取张量的形状
        assert H==self.img_size[0] and W==self.img_size[1],\
        f"输入图像的大小{H}*{W}与模型期望大小{self.img_size[0]}*{self.img_size[1]}不匹配"
        #B,3,224,224->B,768,14,14->B,768,196->B,196,768
        x=self.proj(x).flatten(2).transpose(1,2)
        x=self.norm(x)  #归一化层
        return x
    
class Attention(nn.model):
    def __init__(self,
                dim,#输入的token维度，768
                num_heads=8,#注意力头数
                qkv_bais=False,#生成qkv是是否添加偏置
                qk_scale=None,#用于缩放QK系数，若为None，则使用1/sqrt(head_dim),head_dim=dim//num_heads
                atte_drop_ration=0.,#注意力分数的dropout比率，防止过拟合
                proj_drop_ration=0.#最终投影层dropout比例,防止过拟合
                ):
        super().__init__()
        self.num_heads=num_heads
        head_dim=dim//num_heads
        self.scale=qk_scale or head_dim**-0.5
        self.qkv=nn.Linear(dim,dim*3,bias=qkv_bais)#通过全连接层生成QKV，为了并行计算
        self.att_drop=nn.Dropout(atte_drop_ration)
        self.proj_drop=nn.Dropout(proj_drop_ration)
        #将每个head得到的输出进行concat拼接，再进行线性变换映射会原本的嵌入dim
        self.proj=nn.Linear(dim,dim)

    def forward(self,x):
        B,N,C=x.shape   #batch,num_patches+1,dim
        #B,N,C -> B,N,3*C -> B,N,3,num_heads,head_dim -> 3,B,num_heads,N,head_dim
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]  #(B,num_heads,N,head_dim)

        #q,k点积，再除以缩放系数,得到注意力分数
        #Q ,(B,num_heads,N,head_dim)
        #k.transpose(-2,-1),(B,num_heads,N,head_dim)->(B,num_heads,head_dim,N)
        attn=(q @ k.transpose(-2,-1))*self.scale  #(B,num_heads,N,N)
        attn=attn.softmax(dim=-1)   #对每行处理，使得每行和为1
        #注意力权重对v加权求和
        #attn @ v,(B,num_heads,N,N)@(B,num_heads,N,head_dim)->(B,num_heads,N,head_dim)
        #transpose，(B,num_heads,N,head_dim)->(B,N,num_heads,head_dim)
        x=(attn @ v).transpose(1,2).reshape(B,N,C)  #(B,N,C)

        #线性变换映射会原本的嵌入dim
        x.self.proj(x)
        x=self.dropout(x)   #防止过拟合
        return x
    
class Mlp(nn.model):
    def __init__(self,in_features,hidden_features=None,out_feature=None,act_layer=nn.GELU,drop=0.):
        #in_features:输入特征维度
        #hidden_features:隐藏层特征维度,通常为in_features的4倍
        #out_features:通常等于in_features
        super().__init__()
        out_features=out_features or in_features
        hidden_features=hidden_features or in_features
        self.fc1=nn.Linear(in_features,hidden_features)
        self.act=act_layer()
        self.fc2=nn.Linear(hidden_features,out_features)
        self.drop=nn.Dropout(drop)

    def forward(self,x):
        x=self.fc1(x)   #第一个全连接层
        x=self.act(x)   #激活函数
        x=self.drop(x)  #丢弃一定比例的神经元
        x=self.fc2(x)
        x=self.drop(x)
        return x
    
class Block(nn.model):
    def __init__(self,
                dim,#每个token的维度
                num_heads,#多头自注意力的头数
                mlp_ratio=4.,#mlp隐藏层特征维度是输入特征维度的4倍
                qkv_bias=False,
                qkv_scale=None,
                drop_ration=0., #多头自注意力机制最后的linear后使用的dropout
                att_drop_ration=0., #生成qkv后的dropout比率
                drop_path_ration=0., #drop_path的比例
                act_layer=nn.GELU,  #激活函数
                norm_layer=nn.LayerNorm #归一化层,正则化层
                ):
        super(Block,self).__init__()    #调用父类的构造函数
        self.norm1=norm_layer(dim)  #transformer encoder bloack的第一个layer norm
        #实例化多头自注意力机制
        self.attn=Attention(dim,num_heads=num_heads,qkv_bias=qkv_bias,qkv_scale=qkv_scale,
                            att_drop_ration=att_drop_ration,proj_drop_ration=drop_ration)
        #如果drop_path_ration>0，则使用DropPath，否则使用nn.Identity()不更改
        self.drop_path=DropPath(drop_path_ration) if drop_path_ration>0. else nn.Identity()
        self.norm2=norm_layer(dim)  #transformer encoder bloack的第二个layer norm层
        #实例化MLP层，
        mlp_hidden_dim=int(dim*mlp_ratio)
        self.mlp=Mlp(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop_ration)

    def forward(self,x):
        #前向传播，x先经过layernorm再经过mutiheadattention，再经过drop_path
        x=x+self.drop_path(self.attn(self.norm1(x)))
        #经过layernorm2，再经过mlp，最后再经过drop_path
        x=x+self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class VisionTransformer(nn.model):
    def __init__(self,
                img_size=224,
                patch_size=16,
                in_chans=3,
                num_classes=1000,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.,#mlp隐藏层特征维度是输入特征维度的4倍
                qkv_bias=True,
                qkv_scale=None,
                representation_size=None,
                distilled=False,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.,#drop_path的比例
                embed_layer=PatchEmbed,
                norm_layer=nn.LayerNorm,
                act_layer=None,
                ):
        super(VisionTransformer,self).__init__()
        self.num_classes=num_classes
        self.num_features=self.embed_dim=embed_dim  #embed_dim赋值给
        self.num_tokens=2 if distilled else 1   #num_tokens为1
        #设置一个较小的参数防止除0
        norm_layer=norm_layer or partial(nn.LayerNorm,eps=1e-6)
        act_layer=act_layer or nn.GELU()
        self.patch_embed=embed_layer(img_size=img_size,patch_size=patch_size,in_chans=in_chans,embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches    #得到patchs的个数
        #使用nn.Parameter构建可训练的参数，用零矩阵初始化
        self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))
        self.dist_token=nn.Parameter(torch.zeros(1,1,embed_dim)) if distilled else None
        #pos_embed大小与concat后的大小一致 197*768
        self.pos_embed=nn.Parameter(torch.zeros(1,num_patches+self.num_tokens,embed_dim))
        self.pos_drop=nn.Dropout(p=drop_rate)
        #生成drop_path_rate的等差数列,从0到drop_path_rate，等差数列长度为depth
        dpr=[x.item() for x in torch.linspace(0,drop_path_rate,depth)]
        #使用nn.seqential将列表中的所有模块打包为一个整体
        self.block=nn.Sequential(*[
            Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,
                qkv_scale=qkv_scale,drop_ration=drop_rate,att_drop_ration=attn_drop_rate,
                drop_path_ration=dpr[i],norm_layer=norm_layer,act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm=norm_layer(embed_dim) #通过transformer后的最后一个layer norm
        
        if representation_size and not distilled:
            self.has_logits=True
            self.num_features=representation_size
            self.pre_logits=nn.Sequential(OrderedDict([
                ('fc',nn.Linear(embed_dim,representation_size)),
                ('act',nn.Tanh()),
            ]))
        else:
            self.has_logits=False
            self.pre_logits=nn.Identity()   #pre_logits为恒等映射
        self.head=nn.Linear(self.num_features,num_classes) if num_classes>0 else nn.Identity()
        self.head_dist=None
        if distilled:
            self.head_dist=nn.Linear(embed_dim,num_classes) if num_classes>0 else nn.Identity()
        
        #权重初始化
        nn.init.trunc_normal_(self.pos_embed,std=.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token,std=.02)

        nn.init.trunc_normal_(self.cls_token,std=.02)
        self.apply(_init_vit_weights)

    def forward_features(self,x):
        #x:(B,C,H,W)->(B,num_patches,embed_dim)
        x=self.patch_embed(x)
        #1,1,768->B,1,768
        cls_token=self.cls_token.expand(x.shape[0],-1,-1)
        #如果dist——token存在则拼接dist_token和cls_token，否则只拼接cls_token和输入的x
        if self.dist_token is None:
            x=torch.cat((cls_token,x),dim=1)    #B 197 768
        else:
            x=torch.cat((cls_token,self.dist_token.expand(x.shape[0],-1,-1),x),dim=1)

        x=self.pos_drop(x+self.pos_embed)   #加上位置编码
        x=self.block(x) #经过transformer encoder
        x=self.norm(x)  #layer norm
        if self.dist_token is None:
            return self.pre_logits(x[:,0])
        else:
            return x[:,0],x[:,1]
        

    def forward(self,x):
        x=self.forward_features(x)
        if self.head_dist is not None:
            #分别通过head和head_dist进行预测
            x,x_dist=self.head(x[0]),self.head_dist(x[1])
            #如果是训练模式且不是脚本模式
            if self.training and not torch.jit.is_scripting():
                #则返回两个头部的预测结果
                return x,x_dist
        else:
            x=self.head(x)  #最后的linear 全连接层
        return x
    
def _init_vit_weights(module):
    #判断模块model是否是nn.linear
    if isinstance(module,nn.Linear):
        nn.init.trunc_normal_(module.weight,std=.01)
        if module.bias is not None: #如果线性层存在偏置项，对偏置项初始化为0
            nn.init.zeros_(module.bias)

    elif isinstance(module,nn.Conv2d):
        #对卷积层的权重初始化，适用于卷积
        nn.initkaiming_normal_(module.weight,mode='fan_out')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module,nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)    #对层归一化的权重初始化为1


def vit_base_patch16_224(num_classes:int = 1000,pretrained=False):
    model=VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=None,
        num_classes=num_classes
    )
    return model