#导入必须的包
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torchvision import datasets
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from IPython.display import HTML

# ------------------------------------------------杂项------------------------------------------------
# diffusion 超参数
timesteps = 500
beta1 = 1e-4
beta2 = 0.02

# network 超参数
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu')) #设备
n_feat = 64 # 64 个特征图
n_cfeat = 5 # 5 个文本标签
height = 32 # 16x16 图片

# training 超参数
batch_size = 100
n_epoch = 20
lr = 0.001

# 生成图片的超参数
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1 # b_t时一个在 [beta1, beta2] 范围内线性变化的系数序列
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()   # ab_t表示所有a_t相乘
ab_t[0] = 1

#定义路径
now_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = '{0}/images/train'.format(now_dir)
sample_dir = '{0}/images/sample'.format(now_dir)
save_dir = '{0}/models'.format(now_dir)
print("当前目录:{0} \n 训练时产生的图片在:{1} \n 最后采样得到的图片在:{2} \n 模型保存在:{3}".format(now_dir, train_dir, sample_dir, save_dir))

#创建文件夹
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#设置图像处理:
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],
                         std=[0.5,0.5,0.5])
])


# ------------------------------------------------UNet网络结构------------------------------------------------
class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()

        # 检查输入和输出的通道数是否相同
        self.same_channels = in_channels == out_channels

        # 是否使用残差连接
        self.is_res = is_res

        # 第一层卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),   # 3x3 卷积核 步长：1 填充：1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # Gelu激活函数，一种将RELU与dropout思想结合的激活函数，实验证明效果优于Relu
        )

        # 第二层卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),   # 3x3 卷积核 步长：1 填充：1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # Gelu激活函数
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 如果使用残差连接
        if self.is_res:
            # 使用第一层卷积
            x1 = self.conv1(x)

            # 使用第二层卷积
            x2 = self.conv2(x1)

            # 如果输入输出通道数相同，将残差直接相加
            if self.same_channels:
                out = x + x2
            else:
                # 如果不相同 使用一个 1x1 卷积层在残差连接之前匹配通道数
                shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
                out = shortcut(x) + x2
            #print(f"resconv forward: x {x.shape}, x1 {x1.shape}, x2 {x2.shape}, out {out.shape}")

            # 对输出进行归一化处理，将输出的均值归一化到接近于 1，以避免梯度爆炸或梯度消失的问题。这种归一化处理有助于提高网络的稳定性和训练效果。
            return out / 1.414

        # 如果不使用残差连接，直接输出第二层卷积层的输出
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

    # 获取输出通道数的方法
    def get_out_channels(self):
        return self.conv2[0].out_channels

    # 设置输出通道数的方法
    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels


# 构造Unet的上采样过程
class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()

        # 这个过程主要包含一个用于上采样的反卷积层，后接两个残差卷积块
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]

        # 使用这些层构建一个Sequential Model
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        # 将输入张量和残差张量拼接到一起，这时候通道数相加
        x = torch.cat((x, skip), 1)

        # 将拼接起来的张量注入模型，返回输出
        x = self.model(x)
        return x


# 构建Unet的下采样过程
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()

        # 这个过程主要包含残差卷积块，后接一个用于下采样的Pool层
        layers = [ResidualConvBlock(in_channels, out_channels), ResidualConvBlock(out_channels, out_channels),
                  nn.MaxPool2d(2)]

        # 使用这些层构建一个Sequential Model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # 将输入的张量注入模型，返回输出
        return self.model(x)

# 构建Diffusion中很重要的Embedding过程，用于embed时间步，以及后面部分可能出现的条件embed。
class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        这个class定义了一个通用的单层前馈神经网络，用于将维度 input_dim 的输入数据嵌入到维度 emb_dim 的嵌入空间。
        '''
        self.input_dim = input_dim

        # 建一个sequential model
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        # 将输入按照input_dim展开
        x = x.view(-1, self.input_dim)
        # 将展开的模型应用到模型
        return self.model(x)


# 定义Unet主体
class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat, n_cfeat, height):  # 默认参数，定义模型时可修改
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels  # 输入的通道数
        self.n_feat = n_feat  # 中间层的通道数，也是特征图的数量
        self.n_cfeat = n_cfeat  # 文本标签的数量，在条件生成中使模型生成我们想要的图片
        self.h = height  # 假设 h == w. 由于经过两次下采样，必须能被 4 整除, 由于数据长宽为16，取16...

        # 初始化初始卷积层，(3,16,16)-->(64,16,16)
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # 初始化Unet的两次下采样过程
        self.down1 = UnetDown(n_feat, n_feat)  # down1 (64,16,16)-->(64,8,8)
        self.down2 = UnetDown(n_feat, 2 * n_feat)  # down2 (64,8,8)-->(128,4,4)

        # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        # 使用一层全连接网络嵌入时间步和文本标签 S s
        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)

        # 初始化Unet的三次上采样过程
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h // 4, self.h // 4),  # up-sample
            nn.GroupNorm(8, 2 * n_feat),  # normalize
            nn.ReLU(),
            nn.AvgPool2d((2))
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # 初始化最终的卷积层以映射到与输入图像相同数量的通道
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            # 减少特征图的数量   #in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.GroupNorm(8, n_feat),  # Group norm对batch-size不敏感
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),  # 映射到与输入图像相同数量的通道
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : 图片输入
        t : (batch, n_cfeat)      : 时间步
        c : (batch, n_classes)    : 文本标签
        """
        # x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on

        # 输入通过初始卷积层
        x = self.init_conv(x)  # [100,3,16,16]-->[100,64,16,16]
        # 将结果输入到下采样卷积层
        down1 = self.down1(x)  # [100,64,16,16]-->[100,64,8,8]
        down2 = self.down2(down1)  # [100,64,8,8]-->[100,128,4,4]

        # 将特征图转换为向量并激活
        hiddenvec = self.to_vec(down2)  # [100,128,4,4]-->[100,128,1,1]

        # 如果 context_mask == 1，则屏蔽 context
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        # 嵌入文本和时间步
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)  # (batch, 2*n_feat, 1,1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        # print(f"uunet forward: cemb1 {cemb1.shape}. temb1 {temb1.shape}, cemb2 {cemb2.shape}. temb2 {temb2.shape}")

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

# ------------------------------------------------自定义数据集------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, sfilename, lfilename, transform):
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        self.transform = transform
        self.sprites_shape = self.sprites.shape
        self.slabel_shape = self.slabels.shape

    # 返回数据集中有多少张图（必要）
    def __len__(self):
        return len(self.sprites)

    # 在给定的idx下获取图片和标签（必要）
    def __getitem__(self, idx):
        # 将图片和标签作为一个元组返回
        if self.transform:
            image = self.transform(self.sprites[idx])
            label = torch.tensor(self.slabels[idx]).to(torch.int64)
        return (image, label)

# ------------------------------------------------画图工具------------------------------------------------
def unorm(x):
    # unity norm. results in range of [0,1]
    # assume x (h,w,3)
    xmax = x.max((0, 1))
    xmin = x.min((0, 1))
    return (x - xmin) / (xmax - xmin)

def norm_all(store, n_t, n_s):
    # runs unity norm on all timesteps of all samples
    nstore = np.zeros_like(store)
    for t in range(n_t):
        for s in range(n_s):
            nstore[t, s] = unorm(store[t, s])
    return nstore

def norm_torch(x_all):
    # runs unity norm on all timesteps of all samples
    # input is (n_samples, 3,h,w), the torch image format
    x = x_all.cpu().numpy()
    xmax = x.max((2, 3))
    xmin = x.min((2, 3))
    xmax = np.expand_dims(xmax, (2, 3))
    xmin = np.expand_dims(xmin, (2, 3))
    nstore = (x - xmin) / (xmax - xmin)
    return torch.from_numpy(nstore)
# 画图工具函数
def plot_grid(x, n_sample, n_rows, w):
    # x:(n_sample, 3, h, w)
    ncols = n_sample // n_rows
    grid = make_grid(norm_torch(x), nrow=ncols)  # curiously, nrow is number of columns.. or number of items in the row.
    save_image(grid, sample_dir + f"/run_image_w{w}.png")
    print('saved image at ' + sample_dir + f"/run_image_w{w}.png")
    return grid
def plot_sample(x_gen_store, n_sample, nrows, fn, w, save=False):
    ncols = n_sample // nrows
    sx_gen_store = np.moveaxis(x_gen_store, 2, 4)  # change to Numpy image format (h,w,channels) vs (channels,h,w)
    nsx_gen_store = norm_all(sx_gen_store, sx_gen_store.shape[0],
                             n_sample)  # unity norm to put in range [0,1] for np.imshow

    # create gif of images evolving over time, based on x_gen_store
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(ncols, nrows))

    def animate_diff(i, store):
        print(f'gif animating frame {i} of {store.shape[0]}', end='\r')
        plots = []
        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                plots.append(axs[row, col].imshow(store[i, (row * ncols) + col]))
        return plots

    ani = FuncAnimation(fig, animate_diff, fargs=[nsx_gen_store], interval=100, blit=False, repeat=True, repeat_delay=5000,
                        frames=nsx_gen_store.shape[0])
    plt.close()
    if save:
        ani.save(sample_dir + f"/{fn}_w{w}.gif", dpi=100, writer=PillowWriter(fps=10))
        print('saved gif at ' + sample_dir + f"/{fn}_w{w}.mp4")
    return ani

# ------------------------------------------------训练函数------------------------------------------------

# 将图像扰动到指定的噪音水平
def perturb_input(x, t, noise):
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

def Train():
    model.train()
    for ep in range(n_epoch):
        print(f'epoch {ep}')

        # linearly decay learning rate
        ddpm_optim.param_groups[0]['lr'] = lr * (1 - ep / n_epoch)

        pbar = tqdm(dataloader,colour='green')
        for x, _ in pbar:  # x: images
            ddpm_optim.zero_grad()
            x = x.to(device)

            # 生成噪音扰乱数据
            noise = torch.randn_like(x)
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
            x_pert = perturb_input(x, t, noise)
            # 使用模型去预测噪声
            pred_noise = model(x_pert, t / timesteps)
            # loss 是预测的噪声和真实噪声之间的MSE
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()

            ddpm_optim.step()

        # 每过四个epoch保存一次模型
        if ep % 1 == 0:
            torch.save(model.state_dict(), save_dir + f"/model_{ep}.pth")
#------------------------------------------------采样函数------------------------------------------------

# 减去预测的噪声（但添加一些噪声以避免崩溃）
def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

@torch.no_grad()
def sample_ddpm(n_sample, save_rate=20, epoch=0):
    # x_T ~ N(0, 1), 初始化为噪音
    samples = torch.randn(n_sample, 3, height, height).to(device)

    # 数组来保存生成的过程
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # 采样一些随机噪声注回。对于 i = 1，不要添加噪声
        z = torch.randn_like(samples) if i > 1 else 0

        eps = model(samples, t)    # 预测噪声
        samples = denoise_add_noise(samples, i, eps, z)

        if i == 1:
            plot_grid(samples, n_sample, 4, epoch)

        if i % save_rate ==0 or i==timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate

def sample(epoch):
    model.load_state_dict(torch.load(f"{save_dir}/model_{epoch}.pth", map_location=device, weights_only=True))
    model.eval()
    print("Loaded in Model")
    #显示图像
    plt.clf()
    samples, intermediate_ddpm = sample_ddpm(32, epoch = epoch)
    # animation_ddpm = plot_sample(intermediate_ddpm, 32, 4, save_dir, "ani_run", epoch, save=True)


    # HTML(animation_ddpm.to_jshtml())
#------------------------------------------------主函数------------------------------------------------
if __name__ == '__main__':
    # 加载数据集和构建优化器
    model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)
    dataset = datasets.CIFAR10('../data/cifar-10/datasets', train=True, download=True,
                             transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ddpm_optim = optim.Adam(model.parameters(), lr=lr)
    Train()
    # sample(10)
    # sample(30)
    # sample(100)
    # sample(200)
    # sample(499)