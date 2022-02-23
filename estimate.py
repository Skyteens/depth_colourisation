
from ColorizerEstimate import colorizer_init
import glob,torch,cv2
from utils.dataset import make_dataLoader
from utils.utils import lab_to_rgb
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from io import BytesIO

def getImages(imgPath="./test_imgs"):
    imdir = imgPath
    ext = ['png', 'jpg', 'gif']

    test_path = []
    [test_path.extend(glob.glob(imdir + '/*.' + e)) for e in ext]
    return test_path

def estimate_from_folder(model,num=0,imgPath="./test_imgs"):
    test_path = getImages(imgPath)

    if num > len(test_path) -1 :
        print("Image number is greater than in file")
        return 0 
    
    img_path = test_path[num:num+1]
    test_dl = make_dataLoader(batch_size=1,img_path=img_path,split="val",shuffle=False)
    test_data = next(iter(test_dl))

    sizes = test_data['sizes']

    with torch.no_grad():
        model.setup_input(test_data)
        model.forward()
    fake_colour = model.fake_colour.detach()
    L = model.L
    fakes = lab_to_rgb(L,fake_colour)
    original = lab_to_rgb(model.L,model.ab)
    L=L.permute(0, 2, 3, 1).cpu().numpy()
    # fake_resized,BW_resized= [],[]
    # for i in range(len(fakes)):
    #   dim = (sizes[0][i].item(),sizes[1][i].item())
    #   fake_resize = cv2.resize(fakes[i],dim, interpolation = cv2.INTER_AREA)
    #   fake_resized.append(fake_resize)
    #   BW_resize = cv2.resize(L[i],dim, interpolation = cv2.INTER_AREA)
    #   BW_resized.append(BW_resize)


    # pred= fake_resized
    # pred_L = BW_resized
    #return pred,pred_L
      
    dim = (sizes[0][0].item(),sizes[1][0].item())
    fake_resize = cv2.resize(fakes[0],dim, interpolation = cv2.INTER_AREA)

    gray = cv2.cvtColor(fake_resize, cv2.COLOR_BGR2GRAY)
    gray = np.stack((gray,)*3, axis=-1)

    return fake_resize,gray

def estimate_from_url(model,url):

    response = requests.get(url, timeout=30, headers={'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36'})
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img.save("imgs_download/image.png")
    colour,gray = estimate_from_folder(model,num=0,imgPath="./imgs_download")
    return colour,gray
