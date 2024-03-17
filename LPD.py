import torch
import torch.nn as nn
from odl.contrib import torch as odl_torch

class UNetEncoder(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(n_inputs, n_outputs, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.conv2 = nn.Conv2d(n_outputs, n_outputs, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_outputs)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        p = self.pool1(x)

        return x, p

class UNetBottleneck(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(n_inputs, n_outputs, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.conv2 = nn.Conv2d(n_outputs, n_outputs, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_outputs)
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class UNetDecoder(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.relu = nn.ReLU()
        self.upconv1 = nn.ConvTranspose2d(n_inputs, n_outputs, kernel_size=2, stride=2)
        self.bn0 = nn.BatchNorm2d(n_outputs)
        self.conv1 = nn.Conv2d(n_inputs, n_outputs, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.conv2 = nn.Conv2d(n_outputs, n_outputs, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_outputs)
    
    def forward(self, input, con):
        x = self.upconv1(input)
        x = self.bn0(x)
        x = self.relu(x)
        x = torch.cat([x, con], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class UNetOutput(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(n_inputs, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return x

class UNet(nn.Module):
    def __init__(self, n_inputs, n_UNet):
        super().__init__()

        self.n_UNet = n_UNet

        # Encoder
        
        self.encoder_1 = UNetEncoder(n_inputs,32)
        self.encoder_2 = UNetEncoder(32,64)
        if self.n_UNet >= 3:
            self.encoder_3 = UNetEncoder(64,128)
        if self.n_UNet >= 4:
            self.encoder_4 = UNetEncoder(128,256)
        if self.n_UNet >= 5:
            self.encoder_5 = UNetEncoder(256,512)

        # Bottleneck
        self.bottleneck = UNetBottleneck(2**self.n_UNet * 16, 2**self.n_UNet * 32)
        
        # Decoder
        self.decoder_1 = UNetDecoder(64,32)

        self.decoder_2 = UNetDecoder(128,64)

        if self.n_UNet >= 3:
            self.decoder_3 = UNetDecoder(256,128)
        if self.n_UNet >= 4:
            self.decoder_4 = UNetDecoder(512,256)
        if self.n_UNet >= 5:
            self.decoder_5 = UNetDecoder(1024,512)

        # Output
        self.output = UNetOutput(32)
    
    def forward(self, x):
        e1, x = self.encoder_1(x)
        e2, x = self.encoder_2(x)
        if self.n_UNet >= 3:
            e3, x = self.encoder_3(x)
        if self.n_UNet >= 4:
            e4, x = self.encoder_4(x)
        if self.n_UNet >= 5:    
            e5, x = self.encoder_5(x)
        
        x = self.bottleneck(x)

        if self.n_UNet >= 5:
            x = self.decoder_5(x,e5)
        if self.n_UNet >= 4:
            x = self.decoder_4(x,e4)
        if self.n_UNet >= 3:
            x = self.decoder_3(x,e3)

        x = self.decoder_2(x,e2)
        x = self.decoder_1(x,e1)
        x = self.output(x)

        return x

class LPD(nn.Module):
    def __init__(self, n_LPD, n_UNet, forward_proj, back_proj):
        super().__init__()

        self.n_LPD = n_LPD

        # Define Structures
        self.forward_proj_op = odl_torch.OperatorModule(forward_proj)
        self.back_proj_op = odl_torch.OperatorModule(back_proj)

        if n_LPD>=1:
            self.theta_0 = UNet(1,n_UNet)
            self.lambda_0 = UNet(1,n_UNet)

        if n_LPD>=2:
            self.theta_1 = UNet(3,n_UNet)
            self.lambda_1 = UNet(2,n_UNet)
        
        if n_LPD>=3:
            self.theta_2 = UNet(4,n_UNet)
            self.lambda_2 = UNet(3,n_UNet)

        if n_LPD>=4:
            self.theta_3 = UNet(5,n_UNet)
            self.lambda_3 = UNet(4,n_UNet)

        if n_LPD>=5:
            self.theta_4 = UNet(6,n_UNet)
            self.lambda_4 = UNet(5,n_UNet)
        
        if n_LPD>=6:
            print('LPD with more than 5 layers is not implemented.')
    
    def forward(self, input):
        if self.n_LPD>=1:
            g = input
            h_0 = self.theta_0(g)
            f_g0 = self.back_proj_op(h_0)
            f_0 = self.lambda_0(f_g0)

        if self.n_LPD>=2:
            h_f0 = self.forward_proj_op(f_0)
            h_1 = h_0 + self.theta_1(torch.cat([g, h_0, h_f0], dim=1))
            f_g1 = self.back_proj_op(h_1)
            f_1 = f_0 + self.lambda_1(torch.cat([f_0, f_g1], dim=1))

        if self.n_LPD>=3:
            h_f1 = self.forward_proj_op(f_1)
            h_2 = h_1 + self.theta_2(torch.cat([g, h_0, h_1, h_f1], dim=1))
            f_g2 = self.back_proj_op(h_2)
            f_2 = f_1 + self.lambda_2(torch.cat([f_0, f_1, f_g2], dim=1))

        if self.n_LPD>=4:
            h_f2 = self.forward_proj_op(f_2)
            h_3 = h_2 + self.theta_3(torch.cat([g, h_0, h_1, h_2, h_f2], dim=1))
            f_g3 = self.back_proj_op(h_3)
            f_3 = f_2 + self.lambda_3(torch.cat([f_0, f_1, f_2, f_g3], dim=1))

        if self.n_LPD>=5:
            h_f3 = self.forward_proj_op(f_3)
            h_4 = h_3 + self.theta_4(torch.cat([g, h_0, h_1, h_2, h_3, h_f3], dim=1))
            f_g4 = self.back_proj_op(h_4)
            f_4 = f_3 + self.lambda_4(torch.cat([f_0, f_1, f_2, f_3, f_g4], dim=1))

        if self.n_LPD==1:
            return f_0
        
        if self.n_LPD==2:
            return f_1
        
        if self.n_LPD==3:
            return f_2
        
        if self.n_LPD==4:
            return f_3
        
        if self.n_LPD==5:
            return f_4
            
    def debugh0(self, input):
        g = input
        h_0 = self.theta_0(g)

        return h_0
    
    def debugh1(self, input):
        g = input
        h_0 = self.theta_0(g)
        f_g0 = self.back_proj_op(h_0)
        f_0 = self.lambda_0(f_g0)

        h_f0 = self.forward_proj_op(f_0)
        h_1 = h_0 + self.theta_1(torch.cat([g, h_0, h_f0], dim=1))

        return h_1
    
    def debugh2(self, input):
        g = input
        h_0 = self.theta_0(g)
        f_g0 = self.back_proj_op(h_0)
        f_0 = self.lambda_0(f_g0)

        h_f0 = self.forward_proj_op(f_0)
        h_1 = h_0 + self.theta_1(torch.cat([g, h_0, h_f0], dim=1))
        f_g1 = self.back_proj_op(h_1)
        f_1 = f_0 + self.lambda_1(torch.cat([f_0, f_g1], dim=1))

        h_f1 = self.forward_proj_op(f_1)
        h_2 = h_1 + self.theta_2(torch.cat([g, h_0, h_1, h_f1], dim=1))

        return h_2
    
    def debugh3(self, input):
        g = input
        h_0 = self.theta_0(g)
        f_g0 = self.back_proj_op(h_0)
        f_0 = self.lambda_0(f_g0)

        h_f0 = self.forward_proj_op(f_0)
        h_1 = h_0 + self.theta_1(torch.cat([g, h_0, h_f0], dim=1))
        f_g1 = self.back_proj_op(h_1)
        f_1 = f_0 + self.lambda_1(torch.cat([f_0, f_g1], dim=1))

        h_f1 = self.forward_proj_op(f_1)
        h_2 = h_1 + self.theta_2(torch.cat([g, h_0, h_1, h_f1], dim=1))
        f_g2 = self.back_proj_op(h_2)
        f_2 = f_1 + self.lambda_2(torch.cat([f_0, f_1, f_g2], dim=1))

        h_f2 = self.forward_proj_op(f_2)
        h_3 = h_2 + self.theta_3(torch.cat([g, h_0, h_1, h_2, h_f2], dim=1))

        return h_3
    
    def debugh4(self, input):
        g = input
        h_0 = self.theta_0(g)
        f_g0 = self.back_proj_op(h_0)
        f_0 = self.lambda_0(f_g0)

        h_f0 = self.forward_proj_op(f_0)
        h_1 = h_0 + self.theta_1(torch.cat([g, h_0, h_f0], dim=1))
        f_g1 = self.back_proj_op(h_1)
        f_1 = f_0 + self.lambda_1(torch.cat([f_0, f_g1], dim=1))

        h_f1 = self.forward_proj_op(f_1)
        h_2 = h_1 + self.theta_2(torch.cat([g, h_0, h_1, h_f1], dim=1))
        f_g2 = self.back_proj_op(h_2)
        f_2 = f_1 + self.lambda_2(torch.cat([f_0, f_1, f_g2], dim=1))

        h_f2 = self.forward_proj_op(f_2)
        h_3 = h_2 + self.theta_3(torch.cat([g, h_0, h_1, h_2, h_f2], dim=1))
        f_g3 = self.back_proj_op(h_3)
        f_3 = f_2 + self.lambda_3(torch.cat([f_0, f_1, f_2, f_g3], dim=1))

        h_f3 = self.forward_proj_op(f_3)
        h_4 = h_3 + self.theta_4(torch.cat([g, h_0, h_1, h_2, h_3, h_f3], dim=1))

        return h_4
    
    def debugf0(self, input):
        g = input
        h_0 = self.theta_0(g)
        f_g0 = self.back_proj_op(h_0)
        f_0 = self.lambda_0(f_g0)

        return f_0
    
    def debugf1(self, input):
        g = input
        h_0 = self.theta_0(g)
        f_g0 = self.back_proj_op(h_0)
        f_0 = self.lambda_0(f_g0)

        h_f0 = self.forward_proj_op(f_0)
        h_1 = h_0 + self.theta_1(torch.cat([g, h_0, h_f0], dim=1))
        f_g1 = self.back_proj_op(h_1)
        f_1 = f_0 + self.lambda_1(torch.cat([f_0, f_g1], dim=1))

        return f_1
    
    def debugf2(self, input):
        g = input
        h_0 = self.theta_0(g)
        f_g0 = self.back_proj_op(h_0)
        f_0 = self.lambda_0(f_g0)

        h_f0 = self.forward_proj_op(f_0)
        h_1 = h_0 + self.theta_1(torch.cat([g, h_0, h_f0], dim=1))
        f_g1 = self.back_proj_op(h_1)
        f_1 = f_0 + self.lambda_1(torch.cat([f_0, f_g1], dim=1))

        h_f1 = self.forward_proj_op(f_1)
        h_2 = h_1 + self.theta_2(torch.cat([g, h_0, h_1, h_f1], dim=1))
        f_g2 = self.back_proj_op(h_2)
        f_2 = f_1 + self.lambda_2(torch.cat([f_0, f_1, f_g2], dim=1))

        return f_2
    
    def debugf3(self, input):
        g = input
        h_0 = self.theta_0(g)
        f_g0 = self.back_proj_op(h_0)
        f_0 = self.lambda_0(f_g0)

        h_f0 = self.forward_proj_op(f_0)
        h_1 = h_0 + self.theta_1(torch.cat([g, h_0, h_f0], dim=1))
        f_g1 = self.back_proj_op(h_1)
        f_1 = f_0 + self.lambda_1(torch.cat([f_0, f_g1], dim=1))

        h_f1 = self.forward_proj_op(f_1)
        h_2 = h_1 + self.theta_2(torch.cat([g, h_0, h_1, h_f1], dim=1))
        f_g2 = self.back_proj_op(h_2)
        f_2 = f_1 + self.lambda_2(torch.cat([f_0, f_1, f_g2], dim=1))

        h_f2 = self.forward_proj_op(f_2)
        h_3 = h_2 + self.theta_3(torch.cat([g, h_0, h_1, h_2, h_f2], dim=1))
        f_g3 = self.back_proj_op(h_3)
        f_3 = f_2 + self.lambda_3(torch.cat([f_0, f_1, f_2, f_g3], dim=1))

        return f_3
