import torch
from torch import nn, optim
from  skimage.metrics import peak_signal_noise_ratio,structural_similarity
from models.depthEstimate import init_Depth_model,get_depth
from models.models import *

class Colorizer(nn.Module):
    
    def __init__(self,net=None,depth =False,device ="cpu",lr=1e-4,beta1=0.99,beta2=0.999):
        super().__init__()
        self.device = device

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
      pre_depth = self.L.expand(-1, 3, -1, -1)
      self.depth_maps = get_depth(self.depthModel,pre_depth)

    # the * means end of line for unnamed arguments
    def forward(self):
        if self.depth:
          self.fake_colour = self.model(self.L,self.depth_maps)
        else:
          self.fake_colour = self.model(self.L)

    def inference(self, data,has_weights=False):
        with torch.no_grad():
            out = self.forward(data)
            return out


def colorizer_init():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_encoder = ModelBuilder.build_encoder(
    arch="depth")

    net_decoder = ModelBuilder.build_decoder(
        arch="ppm_unet",
        fc_dim=2048,num_class=2,
        output=True)

    generator = Generator(net_encoder, net_decoder)
    net_G = generator.to(device)
    w= "./weights/depthColour.pt"
    net_G.load_state_dict(torch.load(w, map_location=device))

    for m in net_G.modules():
      for child in m.children():
          if type(child) == nn.BatchNorm2d:
              child.track_running_stats = False
              child.running_mean = None
              child.running_var = None
    net_G.eval()
    model = Colorizer(net=net_G,depth=True,device=device)
    print("Model Initialised")
    return model

