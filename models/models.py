import torch
import torch.nn as nn
from models import resnet

BatchNorm2d = nn.BatchNorm2d

class GeneratorBase(nn.Module):
    def __init__(self):
        super().__init__()

    def pixel(self,pred,label):
        _,pred = torch.max(pred,dim=1)
        valid = (label >= 0 ).long()
        acc_sum = torch.sum(valid*(pred==label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

class Generator(GeneratorBase):
    
    def __init__(self,net_enc,net_dec):
        super().__init__()
        self.encoder = net_enc
        self.decoder = net_dec

    # the * means end of line for unnamed arguments
    def forward(self,x,depth=None,*,segSize=None):
      if depth is not None:
        encoder = self.encoder(x,depth,return_feature_maps=True)
      else:
        encoder = self.encoder(x,return_feature_maps=True)

      pred = self.decoder(encoder)
      return pred

class ModelBuilder:
    
    #weight initialisation
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    @staticmethod
    def build_encoder(arch='resnet50dilated', fc_dim=512, weights=''):
        #pretrained = True if len(weights)==0 else False
        pretrained = False

        arch = arch.lower()
        if arch == 'resnet50':
            orig_resnet = resnet.resnet50(pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.resnet50(pretrained=pretrained)
            net_encoder = Resnet(orig_resnet,dilated=True)
        elif arch == "depth":
          orig_resnet = resnet.resnet50(pretrained=pretrained)
          mono_encoder = Resnet(orig_resnet,dilated=True)
          depth_encoder =Resnet(orig_resnet)
          net_encoder = Depth_encoder(mono_encoder,depth_encoder)
        elif arch == "inst_depth":
          orig_resnet = resnet.resnet50(pretrained=pretrained)
          mono_encoder = Resnet(orig_resnet,dilated=True)
          depth_encoder =Resnet(orig_resnet)
          net_encoder = inst_Depth_encoder(mono_encoder,depth_encoder)

        else:
            raise Exception('Architecture undefined!')
        
        #net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch="ppm_deepsup",fc_dim=512,num_class=150,
                        weights='',output=True):
        arch =arch.lower()
        
        if arch == "unet":
            net_decoder = unet(num_class=num_class,fc_dim=fc_dim,output=output)

        elif arch == "ppm_unet":
            net_decoder = PPM_unet(num_class=num_class,fc_dim=fc_dim,output=output)

        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder

###################### models ########################
def conv3x3_bn_relu(in_planes,out_planes,stride=1):

    return nn.Sequential(
        nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False),
        BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )

class Resnet(nn.Module):
    def __init__(self,orig_resnet,dilated = False, dilate_scale=8):
        super().__init__() 

        if dilated:
          from functools import partial

          if dilate_scale ==8:
              orig_resnet.layer3.apply(
                  partial(self._nostride_dilate,dilate=2)
              )
              orig_resnet.layer4.apply(
                  partial(self._nostride_dilate,dilate=4)
              )
          elif dilate_scale ==16:
              orig_resnet.layer4.apply(
                  partial(self._nostride_dilate,dilate=2)
              )

        #self.conv1 = orig_resnet.conv1
        ori_weight = orig_resnet.conv1.weight.data
        new_w = ori_weight.sum(dim=1, keepdim=True)
        self.conv1=nn.Conv2d(1,64,kernel_size=3,stride=2,
                    padding=1,bias=False)

        self.conv1.weight.data = new_w

        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self,m,dilate):
        classname= m.__class__.__name__
        if classname.find("conv") != -1:
            if m.stride == (2,2):
                m.stride =(1,1)
                if m.kernel_size == (3,3):
                    m.dilation = (dilate//2,dilate//2)
                    m.padding = (dilate//2,dilate//2)
            else:
                if m.kernel_size == (3,3):
                    m.dilation = (dilate,dilate)
                    m.padding = (dilate,dilate)

    def forward(self,x,return_feature_maps=False):
        conv_out = []
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x))); conv_out.append(x)
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x)
        x = self.layer2(x); conv_out.append(x)
        x = self.layer3(x); conv_out.append(x)
        x = self.layer4(x); conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x] 

class Depth_encoder(nn.Module):
    def __init__(self,orig_resnet,dil_resnet):
        super().__init__() 

        #depth encoder
        self.conv1_1 = orig_resnet.conv1
        
        self.bn1_1 = orig_resnet.bn1
        self.relu1_1 = orig_resnet.relu1
        self.conv2_1 = orig_resnet.conv2
        self.bn2_1 = orig_resnet.bn2
        self.relu2_1 = orig_resnet.relu2
        self.conv3_1 = orig_resnet.conv3
        self.bn3_1 = orig_resnet.bn3
        self.relu3_1 = orig_resnet.relu3
        
        
        self.layer1_1 = orig_resnet.layer1
        self.layer2_1 = orig_resnet.layer2
        self.layer3_ori = orig_resnet.layer3
        self.layer4_ori = orig_resnet.layer4

        self.maxpool = orig_resnet.maxpool
        
        #BW encoder
        self.conv1 = orig_resnet.conv1

        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
 
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3_dil = dil_resnet.layer3
        self.layer4_dil = dil_resnet.layer4

        #self.fusion1 = nn.Conv2d(512, 512, kernel_size=1,bias=False)
        #self.fusion2 = nn.Conv2d(1024, 1024, kernel_size=1, bias=False)
        self.fusion3 = nn.Conv2d(2048, 2048, kernel_size=1, bias=False)

        self.fuse_depth_ratio = 0.3
    
    def fuse(self,mono,depth):
      ratio = self.fuse_depth_ratio
      mono = mono * (1-ratio)
      depth =depth * ratio
      return torch.add(mono,depth)

    def encoder_part1(self,in_x):
      out = self.relu1(self.bn1(self.conv1(in_x)))
      out = self.relu2(self.bn2(self.conv2(out)))
      out = self.relu3(self.bn3(self.conv3(out)))
      skip1 = out.clone()
      out = self.maxpool(out)

      out = self.layer1(out)
      return out,skip1

    def forward(self,x,x_depth=None,return_feature_maps=False):
        depth_layers = []
        conv_out = []
        
        #x_depth,_ = self.encoder_part1(x_depth)
        
        x_depth = self.relu1_1(self.bn1_1(self.conv1_1(x_depth)))
        x_depth = self.relu2_1(self.bn2_1(self.conv2_1(x_depth)))
        x_depth = self.relu3_1(self.bn3_1(self.conv3_1(x_depth)))
        x_depth= self.maxpool(x_depth)
        x_depth= self.layer1_1(x_depth)
        x_depth = self.layer2_1(x_depth);depth_layers.append(x_depth)
        x_depth = self.layer3_ori(x_depth); depth_layers.append(x_depth)
        x_depth = self.layer4_ori(x_depth); depth_layers.append(x_depth)

        x,skip1 = self.encoder_part1(x)

        conv_out.append(skip1)
        conv_out.append(x)

        x = self.layer2(x)
        d1 = depth_layers[0]
        #d1 = self.fusion1(d1)
        x = self.fuse(x, d1)
        conv_out.append(x)
        
        x = self.layer3_dil(x)
        d2 = depth_layers[1]
        #d2 = self.fusion2(d2)
        x = self.fuse(x, d2)
        conv_out.append(x)
        
        x = self.layer4_dil(x)
        d3 = depth_layers[2]
        #d3 = self.fusion3(d3)
        x = self.fuse(x, d3)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]   

######## Decoders ####################

class unet(nn.Module):
    def __init__(self,num_class=2,fc_dim=2048):
        super().__init__()

        self.upconv1 = nn.ConvTranspose2d(2048,1024, kernel_size=4,stride=2,padding=1,bias=False)
        self.upconv2 = self.upsample_block(2048,512)
        self.upconv3 = self.upsample_block(1024,256)
        self.upconv4 = self.upsample_block(512,128)
        self.upconv5 = self.upsample_block(256,64)
      
        self.lastConv =nn.Conv2d(64,num_class,kernel_size=1)

    def upsample_block(self,out_c,in_c):
        upconv = nn.ConvTranspose2d(out_c,in_c, kernel_size=4,stride=2,padding=1,bias=False)
   

        block = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(out_c,in_c, kernel_size=4,stride=2,padding=1,bias=False),
                nn.BatchNorm2d(in_c)
        )

        return block

    def forward(self,conv_out,segSize=None):
        conv5 = conv_out[-1]

        x= self.upconv1(conv5)
        x = torch.cat([x,conv_out[-2]],1)
        x= self.upconv2(x)
        x = torch.cat([x,conv_out[-3]],1)
        x= self.upconv3(x)
        x = torch.cat([x,conv_out[-4]],1)
        x= self.upconv4(x)
        x = torch.cat([x,conv_out[-5]],1)
        x= self.upconv5(x)
        x= self.lastConv(x)
        return x

class PPM_unet(nn.Module):
    def __init__(self,num_class=2,fc_dim=4096,output=True,pool_scales=(1,2,3,6)):
        super().__init__()
        self.output =output

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim,512,kernel_size=1,bias=False),
                #BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))

        self.ppm = nn.ModuleList(self.ppm)

        self.upconv2 = self.upsample_block(2048,512)
        self.upconv3 = self.upsample_block(1024,256)
        self.upconv4 = self.upsample_block(512,128)
        self.upconv5 = self.upsample_block(256,64)
      
        self.lastConv =nn.Conv2d(64,num_class,kernel_size=1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512,512,kernel_size=3,padding=1,bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        
    def upsample_block(self,out_c,in_c):
        upconv = nn.ConvTranspose2d(out_c,in_c, kernel_size=4,stride=2,padding=1,bias=False)
   

        block = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(out_c,in_c, kernel_size=4,stride=2,padding=1,bias=False),
                nn.BatchNorm2d(in_c)
        )
        return block

    def forward(self,conv_out,segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()

        ppm_out= [conv5]
        
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2],input_size[3]),
                mode ="bilinear", align_corners=False
            ))
        
        ppm_out = torch.cat(ppm_out,1)

        x=self.conv_last(ppm_out)

        if self.output:
            x = nn.functional.interpolate(x,size=(32,32),mode="bilinear",align_corners=False)

            x = torch.cat([x,conv_out[-3]],1)
            x= self.upconv3(x)
            x = torch.cat([x,conv_out[-4]],1)
            x= self.upconv4(x)
            x = torch.cat([x,conv_out[-5]],1)
            x= self.upconv5(x)
            x= self.lastConv(x)
        return x

def main():
    net_encoder = ModelBuilder.build_encoder(
    arch="depth")

    net_decoder = ModelBuilder.build_decoder(
        arch="ppm_unet",
        fc_dim=2048,num_class=2,
        output=True)


    model = Generator(net_encoder, net_decoder)

    x = torch.zeros(1, 1, 256, 256, dtype=torch.float, requires_grad=False)

    out = model(x,depth = x)
    print(out.shape)

############ END ####################
if __name__ == "__main__":

    main()
       
    """     net_encoder = ModelBuilder.build_encoder(
        arch="resnet50dilated")

    net_decoder = ModelBuilder.build_decoder(
        arch="ppm_unet",
        fc_dim=2048,num_class=2,
        output=True)

    x = torch.zeros(16, 1, 256, 256, dtype=torch.float, requires_grad=False)

    Q = [
    torch.zeros(16, 128,128,128, dtype=torch.float, requires_grad=False),
    torch.zeros(16, 256, 64, 64, dtype=torch.float, requires_grad=False),
    torch.zeros(16, 512, 32, 32, dtype=torch.float, requires_grad=False),
    torch.zeros(16, 1024, 16, 16, dtype=torch.float, requires_grad=False),
    torch.zeros(16, 2048,8,8, dtype=torch.float, requires_grad=False)
    ]
    out = net_decoder(Q,segSize = (256,256))
    #out = net_encoder(x,return_feature_maps=True)
    print(out.shape) """