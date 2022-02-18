from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from PIL import Image

class ColorDataset(Dataset):
    def __init__(self,img_path,split='train'):
        
        self.size = 256
        self.split =split
        self.img_path =img_path
        
        if self.split == 'train':
            self.transforms = transforms.Compose(
                [transforms.Resize((self.size, self.size),  transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                ])
        elif self.split == 'val':
            self.transforms = transforms.Resize((self.size, self.size), transforms.InterpolationMode.BICUBIC)

      
            
    def __getitem__(self,idx):
        img = Image.open(self.img_path[idx]).convert("RGB")
        width, height = img.size
        imgsize = (width, height)
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype('float32')
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0],...]/50. -1.
        ab = img_lab[[1,2],...]/110.
        
        return {'L': L, 'ab': ab ,'sizes':imgsize}
    
    def __len__(self):
        return len(self.img_path)
    
    
def make_dataLoader(batch_size=16, n_workers=4,pin_memory=True,shuffle=True,**kwargs):
    dataset = ColorDataset(**kwargs)
    dataloader = DataLoader(dataset,batch_size=batch_size,num_workers=n_workers,pin_memory=pin_memory,shuffle=shuffle)
    
    return dataloader