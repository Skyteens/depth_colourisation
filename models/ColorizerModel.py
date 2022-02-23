import torch
from torch import nn, optim
from  skimage.metrics import peak_signal_noise_ratio,structural_similarity
from .depthEstimate import init_Depth_model,get_depth

class Colorizer(nn.Module):
    
    def __init__(self,net=None,depth =False,lr=1e-4,beta1=0.99,beta2=0.999):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.model = net.to(self.device)
        self.depth =depth
        self.criterion = nn.SmoothL1Loss() 

        
        if self.depth:
          self.depthModel = init_Depth_model()
        
        self.opt = optim.Adam(self.model.parameters(),lr=lr,betas=(beta1,beta2))


    def setup_input(self,data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        if self.depth:
          self.get_depthMaps(data)
      
    def get_depthMaps(self,data):
      # rgb = data['depthMap'].to(self.device)
      # rgb=rgb.permute(0,3,1,2).float()
      pre_depth = self.L.expand(-1, 3, -1, -1)
      self.depth_maps = get_depth(self.depthModel,pre_depth)

    # the * means end of line for unnamed arguments
    def forward(self):
        if self.depth:
          self.fake_colour = self.model(self.L,self.depth_maps)
        else:
          self.fake_colour = self.model(self.L)
    
    def optimise(self):
        self.forward()
        self.model.train()
        self.loss = self.criterion(self.fake_colour,self.ab)
        self.opt.zero_grad()
        self.loss.backward()
        self.opt.step()

    def valuate(self):
        with torch.no_grad():
          self.forward()
          real = lab_to_rgb(self.L.detach(),self.ab.detach())
          fake = lab_to_rgb(self.L.detach(),self.fake_colour.detach())
          self.psnr = peak_signal_noise_ratio(real,fake)
          self.ssim = structural_similarity(real,fake,multichannel=True)

    def inference(self, data,has_weights=False):
        with torch.no_grad():
            out = self.forward(data)
            return out