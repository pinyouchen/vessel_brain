
import numpy as np

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, num_layers=4, max_nn=512, \
                down_min_dropout = 0.1, down_max_dropout = 0.6, up_min_dropout = 0.1, up_max_dropout = 0.5):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.num_layers = num_layers
        self.down_min_dropout = down_min_dropout
        self.down_max_dropout = down_max_dropout
        self.up_min_dropout = up_min_dropout
        self.up_max_dropout = up_max_dropout

        self.down_layers = nn.ModuleList()
        self.down_dropout_layers = nn.ModuleList()

        self.up_layers = nn.ModuleList()
        self.up_dropout_layers = nn.ModuleList()

        # Define progressive Dropout rates for down and up layers
        self.down_dropout_rates = np.linspace(self.down_min_dropout, self.down_max_dropout, self.num_layers).tolist()  # Down 使用更强的正则化
        self.up_dropout_rates = np.linspace(self.up_max_dropout, self.up_min_dropout, self.num_layers).tolist()    # Up 使用较低的正则化

        inc_out_nn = (max_nn // (2 ** self.num_layers))
        self.inc = (DoubleConv(n_channels, inc_out_nn))
        outc_in_nn = (max_nn // (2 ** self.num_layers))
        self.outc = (OutConv(outc_in_nn, n_classes))

        factor = 2 if bilinear else 1

        # Downsampling layers
        for i in range(self.num_layers):
            down_in_nn = max_nn // (2 ** (self.num_layers - i))
            down_out_nn = max_nn // (2 ** ((self.num_layers - i - 1))) // factor if i == (self.num_layers - 1) else max_nn // (2 ** ((self.num_layers - i - 1)))

            down = Down(down_in_nn, down_out_nn)
            down_dropout = nn.Dropout(self.down_dropout_rates[i])

            self.down_layers.append(down)
            self.down_dropout_layers.append(down_dropout)

        # Upsampling layers
        for i in range(self.num_layers):
            up_in_nn = max_nn // (2 ** i)
            up_out_nn = max_nn // (2 ** (i + 1)) if i == (self.num_layers - 1) else max_nn // (2 ** (i + 1)) // factor

            up = Up(up_in_nn, up_out_nn, bilinear)
            up_dropout = nn.Dropout(self.up_dropout_rates[i])

            self.up_layers.append(up)
            self.up_dropout_layers.append(up_dropout)

        # Add classifier for branch decision
        self.classifier = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 2)  # 二分类：全黑或非全黑
        )

    def forward(self, x):
        # Classification branch to decide if the input is all black
        black_flag = self.classifier(x)

        current_out = self.inc(x)
        down_output_list = []

        # Downsampling
        for i in range(self.num_layers):
            down_output_list.append(current_out)
            current_out = self.down_layers[i](current_out)
            current_out = self.down_dropout_layers[i](current_out)

        # Upsampling
        for i in range(self.num_layers):
            current_out = self.up_layers[i](current_out, down_output_list[self.num_layers - i - 1])
            current_out = self.up_dropout_layers[i](current_out)

        out = self.outc(current_out)
        return out, black_flag
    
    def freeze_unet(self):
        """冻结 UNet 主体的权重"""
        for param in self.inc.parameters():
            param.requires_grad = False
        for layer in self.down_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in self.up_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.outc.parameters():
            param.requires_grad = False

    def unfreeze_unet(self):
        """解冻 UNet 主体的权重"""
        for param in self.inc.parameters():
            param.requires_grad = True
        for layer in self.down_layers:
            for param in layer.parameters():
                param.requires_grad = True
        for layer in self.up_layers:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.outc.parameters():
            param.requires_grad = True

class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, num_layers=4, max_nn=512, \
                down_min_dropout = 0.2, down_max_dropout = 0.6, up_min_dropout = 0.2, up_max_dropout = 0.5):
        super(UNetPlusPlus, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.num_layers = num_layers
        self.down_min_dropout = down_min_dropout
        self.down_max_dropout = down_max_dropout
        self.up_min_dropout = up_min_dropout
        self.up_max_dropout = up_max_dropout

        self.down_layers = nn.ModuleList()
        self.down_dense_layers = nn.ModuleList()
        self.down_se_layers = nn.ModuleList()
        self.down_dropout_layers = nn.ModuleList()

        self.up_layers = nn.ModuleList()
        self.up_cbam_layers = nn.ModuleList()
        self.up_dropout_layers = nn.ModuleList()

        # Define progressive Dropout rates for down and up layers
        self.down_dropout_rates = np.linspace(self.down_min_dropout, self.down_max_dropout, self.num_layers).tolist()  # Down 使用更强的正则化
        self.up_dropout_rates = np.linspace(self.up_max_dropout, self.up_min_dropout, self.num_layers).tolist()    # Up 使用较低的正则化

        inc_out_nn = (max_nn // (2 ** self.num_layers))
        self.inc = (DoubleConv(n_channels, inc_out_nn))
        self.dense_in = DenseBlock(inc_out_nn, growth_rate=8, n_layers=1)
        outc_in_nn = (max_nn // (2 ** self.num_layers))
        self.outc = (OutConv(outc_in_nn, n_classes))

        factor = 2 if bilinear else 1

        # Downsampling layers
        for i in range(self.num_layers):
            down_in_nn = max_nn // (2 ** (self.num_layers - i))
            down_out_nn = max_nn // (2 ** ((self.num_layers - i - 1))) // factor if i == (self.num_layers - 1) else max_nn // (2 ** ((self.num_layers - i - 1)))

            down = Down(down_in_nn, down_out_nn)
            down_se = SEBlock(down_out_nn)
            down_dropout = nn.Dropout(self.down_dropout_rates[i])

            self.down_layers.append(down)
            self.down_se_layers.append(down_se)
            self.down_dropout_layers.append(down_dropout)

            if i < (self.num_layers - 1):
                down_dense = DenseBlock(down_out_nn, growth_rate=8, n_layers=2)
                self.down_dense_layers.append(down_dense)

        # Upsampling layers
        for i in range(self.num_layers):
            up_in_nn = max_nn // (2 ** i)
            up_out_nn = max_nn // (2 ** (i + 1)) if i == (self.num_layers - 1) else max_nn // (2 ** (i + 1)) // factor

            up = Up(up_in_nn, up_out_nn, bilinear)
            up_cbam = CBAMBlock(up_out_nn)
            up_dropout = nn.Dropout(self.up_dropout_rates[i])

            self.up_layers.append(up)
            self.up_cbam_layers.append(up_cbam)
            self.up_dropout_layers.append(up_dropout)

        # Add classifier for branch decision
        self.classifier = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 2)  # 二分类：全黑或非全黑
        )

    def forward(self, x):
        # Classification branch to decide if the input is all black
        black_flag = self.classifier(x)

        current_out = self.inc(x)
        current_out = self.dense_in(current_out)
        down_output_list = []

        # Downsampling
        for i in range(self.num_layers):
            down_output_list.append(current_out)
            current_out = self.down_layers[i](current_out)
            if i < (self.num_layers - 1):
                current_out = self.down_dense_layers[i](current_out)
            current_out = self.down_se_layers[i](current_out)
            current_out = self.down_dropout_layers[i](current_out)

        # Upsampling
        for i in range(self.num_layers):
            processed_down_output = self.up_cbam_layers[i](down_output_list[self.num_layers - i - 1])
            current_out = self.up_layers[i](current_out, processed_down_output)
            current_out = self.up_dropout_layers[i](current_out)

        out = self.outc(current_out)
        return out, black_flag

    def freeze_unet(self):
        """冻结 UNet 主体的权重"""
        for param in self.inc.parameters():
            param.requires_grad = False
        for layer in self.down_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in self.up_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.outc.parameters():
            param.requires_grad = False

    def unfreeze_unet(self):
        """解冻 UNet 主体的权重"""
        for param in self.inc.parameters():
            param.requires_grad = True
        for layer in self.down_layers:
            for param in layer.parameters():
                param.requires_grad = True
        for layer in self.up_layers:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.outc.parameters():
            param.requires_grad = True

class UNet_Residual(nn.Module):
    """
    一個使用 ResidualBlock 取代傳統 DoubleConv 的 U-Net 實作。
    - 結構上與 UNet 類似，但在 inc (最前端)、Down 區塊、Up 區塊中使用殘差塊。
    - 依照需求，你也可以選擇只在 Encoder 或 Decoder 使用 ResidualBlock。
    """

    def __init__(self, 
                 n_channels,
                 n_classes,
                 bilinear=False,
                 num_layers=4,
                 max_nn=512,
                 down_min_dropout=0.1,
                 down_max_dropout=0.6,
                 up_min_dropout=0.1,
                 up_max_dropout=0.5):
        super(UNet_Residual, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.num_layers = num_layers

        self.down_min_dropout = down_min_dropout
        self.down_max_dropout = down_max_dropout
        self.up_min_dropout   = up_min_dropout
        self.up_max_dropout   = up_max_dropout

        self.down_layers = nn.ModuleList()
        self.down_dropout_layers = nn.ModuleList()

        self.up_layers = nn.ModuleList()
        self.up_dropout_layers = nn.ModuleList()

        # 漸進式 Dropout 率
        self.down_dropout_rates = np.linspace(
            self.down_min_dropout, 
            self.down_max_dropout, 
            self.num_layers
        ).tolist()

        self.up_dropout_rates = np.linspace(
            self.up_max_dropout, 
            self.up_min_dropout, 
            self.num_layers
        ).tolist()

        # factor 用來決定是否在最底層減半channel (bilinear=True 時)
        factor = 2 if bilinear else 1

        # -------------------------
        # 1) inc (前置殘差塊)
        # -------------------------
        # 與原本 UNet 的 inc_out_nn = max_nn // (2^num_layers) 相同
        inc_out_nn = max_nn // (2 ** self.num_layers)
        self.inc = ResidualBlock(in_channels=n_channels, out_channels=inc_out_nn)

        # -------------------------
        # 2) outc (最後輸出卷積)
        # -------------------------
        outc_in_nn = max_nn // (2 ** self.num_layers)
        self.outc = OutConv(outc_in_nn, n_classes)

        # -------------------------
        # 3) Downsampling layers
        # -------------------------
        # 取代原本 (MaxPool + DoubleConv) 為 (MaxPool + ResidualBlock)
        for i in range(self.num_layers):
            down_in_nn = max_nn // (2 ** (self.num_layers - i))
            if i == (self.num_layers - 1):
                down_out_nn = (max_nn // (2 ** (self.num_layers - i - 1))) // factor
            else:
                down_out_nn = max_nn // (2 ** (self.num_layers - i - 1))

            # 用一個簡單的下采樣區塊：MaxPool2d + ResidualBlock
            down_block = nn.Sequential(
                nn.MaxPool2d(2),
                ResidualBlock(down_in_nn, down_out_nn)
            )
            down_dropout = nn.Dropout(self.down_dropout_rates[i])

            self.down_layers.append(down_block)
            self.down_dropout_layers.append(down_dropout)

        # -------------------------
        # 4) Upsampling layers
        # -------------------------
        # 仍然使用原先的 Up 類別 (上采樣 + concat + DoubleConv)
        # 只是在 "DoubleConv" 那段，我們可以改成一個 "ResidualBlock"
        # 為了簡化，這裡直接繼承原本的 Up，但要在 Up 中改用 ResidualBlock。
        # 如果你想要做更「純」的殘差式 Up，可以自定義一個 UpResidual。
        for i in range(self.num_layers):
            up_in_nn = max_nn // (2 ** i)
            if i == (self.num_layers - 1):
                up_out_nn = max_nn // (2 ** (i + 1))
            else:
                up_out_nn = (max_nn // (2 ** (i + 1))) // factor

            # 這裡我們直接用原本的 Up 類別，它裡面默認用 DoubleConv，
            # 如需完全殘差化，可以把 Up 改成 UpResidual (見下方做法)。
            up_layer = Up(up_in_nn, up_out_nn, bilinear=bilinear)

            up_dropout = nn.Dropout(self.up_dropout_rates[i])
            self.up_layers.append(up_layer)
            self.up_dropout_layers.append(up_dropout)

        # -------------------------
        # 5) 額外的分類分支 (判斷是否全黑)
        # -------------------------
        self.classifier = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 2)  # 二分類: 全黑 or 非全黑
        )

    def forward(self, x):
        # 1) 全黑判斷 (與原始 UNet 相同)
        black_flag = self.classifier(x)

        # 2) inc
        current_out = self.inc(x)

        # 3) Downsampling
        down_output_list = []
        for i in range(self.num_layers):
            down_output_list.append(current_out)
            current_out = self.down_layers[i](current_out)   # (MaxPool2d + ResidualBlock)
            current_out = self.down_dropout_layers[i](current_out)

        # 4) Upsampling
        for i in range(self.num_layers):
            # skip connection 由 down_output_list 中取出
            skip_feat = down_output_list[self.num_layers - i - 1]
            # Up(...): x1 = current_out, x2 = skip_feat
            current_out = self.up_layers[i](current_out, skip_feat)
            current_out = self.up_dropout_layers[i](current_out)

        # 5) outc
        out = self.outc(current_out)
        return out, black_flag

    def freeze_unet(self):
        """冻结 UNet 主體的權重 (inc, down_layers, up_layers, outc)"""
        for param in self.inc.parameters():
            param.requires_grad = False
        for block in self.down_layers:
            for param in block.parameters():
                param.requires_grad = False
        for layer in self.up_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.outc.parameters():
            param.requires_grad = False

    def unfreeze_unet(self):
        """解冻 UNet 主體的權重"""
        for param in self.inc.parameters():
            param.requires_grad = True
        for block in self.down_layers:
            for param in block.parameters():
                param.requires_grad = True
        for layer in self.up_layers:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.outc.parameters():
            param.requires_grad = True


class UNet_Attention(nn.Module):
    """
    修正版的單一類別 Attention U-Net：
      1) 在 Up 模組中，若使用 ConvTranspose2d，改為 (in_channels -> out_channels)，
         避免出現「期望 256，實際 512」的錯誤。
      2) 在 AttentionBlock forward 時，如 gating 與 skip 的空間維度不同，則用 F.interpolate 對齊。
      3) 其餘結構 (DoubleConv, Down, freeze/unfreeze, Classifier etc.) 保持原邏輯。
    """

    # --------------------------------------------------
    # (1) DoubleConv
    # --------------------------------------------------
    class DoubleConv(nn.Module):
        """(Conv => BN => ReLU) * 2"""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.double_conv(x)

    # --------------------------------------------------
    # (2) Down
    # --------------------------------------------------
    class Down(nn.Module):
        """下採樣：MaxPool(2) -> DoubleConv"""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.down = nn.Sequential(
                nn.MaxPool2d(2),
                UNet_Attention.DoubleConv(in_channels, out_channels)
            )

        def forward(self, x):
            return self.down(x)

    # --------------------------------------------------
    # (3) Up
    # --------------------------------------------------
    class Up(nn.Module):
        """
        上採樣 -> (ConvTranspose2d 或 Upsample) -> concat -> DoubleConv
        修正：若 bilinear=False，ConvTranspose2d(in_channels -> out_channels)。
        """
        def __init__(self, in_channels, out_channels, bilinear=True):
            super().__init__()
            if bilinear:
                # 雙線性插值不改變通道
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                # concat 後 channels = in_channels + skip_channels => 之後 DoubleConv
                # 需要 self.conv = DoubleConv(in_channels=?, out_channels=?)
                # 但我們在 init 時由外部指定 in_channels、out_channels
            else:
                # 轉置卷積：輸入 in_channels -> 輸出 out_channels
                self.up = nn.ConvTranspose2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=2, 
                    stride=2
                )
            # 如果使用 bilinear，上採樣後 x1 仍是 in_channels，
            # concat(skip_channels + x1_channels) => 2*out_channels => DoubleConv(2*out_channels, out_channels)
            # 如果使用轉置卷積，上採樣後 x1 會變成 out_channels，
            # concat(skip_channels + out_channels) => ?

            # 這裡為簡化，統一採用 "DoubleConv( out_channels * 2, out_channels )"
            self.conv = UNet_Attention.DoubleConv(out_channels * 2, out_channels)

        def forward(self, x1, x2):
            """
            x1: decoder feature (channels=?)
            x2: encoder skip feature (channels=out_channels)，已經注意力處理
            """
            x1 = self.up(x1)
            # 若大小不一致，做 padding
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            # concat => channel = out_channels + out_channels = 2*out_channels
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
            return x

    # --------------------------------------------------
    # (4) OutConv
    # --------------------------------------------------
    class OutConv(nn.Module):
        """最後輸出 1x1 Conv"""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    # --------------------------------------------------
    # (5) AttentionBlock
    # --------------------------------------------------
    class AttentionBlock(nn.Module):
        """
        Attention Gate:
        g: decoder feature (gating)
        x: encoder skip feature
        """
        def __init__(self, in_g, in_x, mid_channels):
            super().__init__()
            self.W_g = nn.Sequential(
                nn.Conv2d(in_g, mid_channels, kernel_size=1),
                nn.BatchNorm2d(mid_channels)
            )
            self.W_x = nn.Sequential(
                nn.Conv2d(in_x, mid_channels, kernel_size=1),
                nn.BatchNorm2d(mid_channels)
            )
            self.psi = nn.Sequential(
                nn.Conv2d(mid_channels, 1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            # 先確保 g, x 的空間大小一致，若不一致就對 g 做插值
            if g.size()[2:] != x.size()[2:]:
                g = F.interpolate(g, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            out = x * psi  # alpha * skip
            return out

    # --------------------------------------------------
    # 主體 UNet_Attention
    # --------------------------------------------------
    def __init__(
        self,
        n_channels,
        n_classes,
        bilinear=False,
        num_layers=4,
        max_nn=512,
        down_min_dropout=0.1,
        down_max_dropout=0.6,
        up_min_dropout=0.1,
        up_max_dropout=0.5
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.num_layers = num_layers
        self.down_min_dropout = down_min_dropout
        self.down_max_dropout = down_max_dropout
        self.up_min_dropout   = up_min_dropout
        self.up_max_dropout   = up_max_dropout

        self.down_layers = nn.ModuleList()
        self.down_dropout_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.up_dropout_layers = nn.ModuleList()

        # Attention Blocks
        self.attn_blocks = nn.ModuleList()

        # 漸進式 Dropout 率
        self.down_dropout_rates = np.linspace(self.down_min_dropout, self.down_max_dropout, self.num_layers).tolist()
        self.up_dropout_rates   = np.linspace(self.up_max_dropout,   self.up_min_dropout,   self.num_layers).tolist()

        factor = 2 if bilinear else 1

        # -----------------------
        # 1) inc
        # -----------------------
        inc_out_nn = max_nn // (2 ** self.num_layers)
        self.inc = UNet_Attention.DoubleConv(n_channels, inc_out_nn)

        # -----------------------
        # 2) outc
        # -----------------------
        outc_in_nn = max_nn // (2 ** self.num_layers)
        self.outc = UNet_Attention.OutConv(outc_in_nn, n_classes)

        # -----------------------
        # Downsampling (Encoder)
        # -----------------------
        self.down_out_channels = []
        for i in range(self.num_layers):
            down_in_nn = max_nn // (2 ** (self.num_layers - i))
            if i == (self.num_layers - 1):
                down_out_nn = (max_nn // (2 ** (self.num_layers - i - 1))) // factor
            else:
                down_out_nn = max_nn // (2 ** (self.num_layers - i - 1))

            down = UNet_Attention.Down(down_in_nn, down_out_nn)
            down_dropout = nn.Dropout(self.down_dropout_rates[i])

            self.down_layers.append(down)
            self.down_dropout_layers.append(down_dropout)
            self.down_out_channels.append(down_out_nn)

        # -----------------------
        # Upsampling (Decoder) + AttentionBlock
        # -----------------------
        for i in range(self.num_layers):
            # 這層 decoder 的輸入通道
            # e.g. i=0 => up_in_nn = max_nn
            up_in_nn = max_nn // (2 ** i)

            # 這層 decoder 的輸出通道
            if i == (self.num_layers - 1):
                up_out_nn = max_nn // (2 ** (i + 1))
            else:
                up_out_nn = (max_nn // (2 ** (i + 1))) // factor

            up = UNet_Attention.Up(up_in_nn, up_out_nn, bilinear=self.bilinear)
            up_dropout = nn.Dropout(self.up_dropout_rates[i])
            self.up_layers.append(up)
            self.up_dropout_layers.append(up_dropout)

            # Attention Block (先用 placeholder，forward 時動態 new)
            attn_block = UNet_Attention.AttentionBlock(
                in_g=up_in_nn,
                in_x=up_in_nn,
                mid_channels=max(1, up_in_nn // 2)
            )
            self.attn_blocks.append(attn_block)

        # -----------------------
        # 額外分支：Classifier (判斷是否全黑)
        # -----------------------
        self.classifier = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 2)  # 二分類: 全黑 or 非全黑
        )

    def forward(self, x):
        # 1) 是否全黑判斷
        black_flag = self.classifier(x)

        # 2) inc
        current_out = self.inc(x)
        down_output_list = []

        # 3) Downsampling
        for i in range(self.num_layers):
            down_output_list.append(current_out)
            current_out = self.down_layers[i](current_out)
            current_out = self.down_dropout_layers[i](current_out)

        # 4) Up + Attention
        for i in range(self.num_layers):
            skip_id = self.num_layers - i - 1
            skip_feat = down_output_list[skip_id]

            # gating_ch = current_out.shape[1]
            # skip_ch   = skip_feat.shape[1]
            # mid_ch    = max(1, gating_ch // 2)
            # 重新生成 AttentionBlock
            gating_ch = current_out.shape[1]
            skip_ch   = skip_feat.shape[1]
            attn_block = self.AttentionBlock(
                in_g=gating_ch,
                in_x=skip_ch,
                mid_channels=max(1, gating_ch // 2)
            ).to(current_out.device)

            # 做 Attention
            attn_skip = attn_block(current_out, skip_feat)

            # Up
            current_out = self.up_layers[i](current_out, attn_skip)
            current_out = self.up_dropout_layers[i](current_out)

        # 5) 最後輸出
        out = self.outc(current_out)
        return out, black_flag

    def freeze_unet(self):
        """凍結 UNet 主體權重"""
        for param in self.inc.parameters():
            param.requires_grad = False
        for down_block in self.down_layers:
            for param in down_block.parameters():
                param.requires_grad = False
        for up_block in self.up_layers:
            for param in up_block.parameters():
                param.requires_grad = False
        for param in self.outc.parameters():
            param.requires_grad = False
        for param in self.attn_blocks.parameters():
            param.requires_grad = False

    def unfreeze_unet(self):
        """解凍 UNet 主體權重"""
        for param in self.inc.parameters():
            param.requires_grad = True
        for down_block in self.down_layers:
            for param in down_block.parameters():
                param.requires_grad = True
        for up_block in self.up_layers:
            for param in up_block.parameters():
                param.requires_grad = True
        for param in self.outc.parameters():
            param.requires_grad = True
        for param in self.attn_blocks.parameters():
            for param in self.attn_blocks.parameters():
                param.requires_grad = True