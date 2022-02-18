import torch
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from .utils import *
import glob 

def visualise(model,data,save=True):
    # for m in model.modules():
    #   for child in m.children():
    #       if type(child) == nn.BatchNorm2d:
    #           child.track_running_stats = False
    #           child.running_mean = None
    #           child.running_var = None
    #model.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.train()
    fake_colour = model.fake_colour.detach()
    real_colour =model.ab
    L= model.L
    fake_imgs = lab_to_rgb(L,fake_colour)
    real_imgs = lab_to_rgb(L,real_colour)
    
    fig= plt.figure(figsize=(15,8))
    
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")

def train_model(model,train_dl,epochs,vis_every=400):
    val_data = next(iter(val_dl))
    for e in range(epochs):
        loss_meter_dict = create_loss_meters()
        loss_i = 0
 
        for data in tqdm(train_dl):
            model.setup_input(data)
            model.optimise()
            update_losses(model, loss_meter_dict, count=data['L'].size(0))
            
            loss_i += 1
            if loss_i % vis_every == 0:
                print(f"\n Epoch {e+1}/{epochs}")
                print(f"Iteration {loss_i}/{len(train_dl)}")
                log_results(loss_meter_dict)
                visualise(model,val_data,save=False)
                torch.save(model.model.state_dict(), f"test_model-{e}.pt")

def make_training_data(use_colab=False,coco_path=None):
    if use_colab == True:
        path = coco_path
    else:
        path = glob.glob("./dataset/coco/*.jpg")# Grabbing all the image file names

    img_paths =  glob.glob(path + "/*.jpg")# All paths to COCO dataset images
    np.random.seed(16) # Seeding for reproducible results
    print(len(img_paths))
    paths_subset = np.random.choice(img_paths, 20_000, replace=False) # Randomly choosing 10,000 images
    rand_idxs = np.random.permutation(20_000) # Shuffling the indexes
    train_idxs = rand_idxs[:16000] # Using first 8000 images for training
    val_idxs = rand_idxs[16000:] # Using last 2000 images for validating
    train_paths = paths_subset[train_idxs] 
    val_paths = paths_subset[val_idxs]
    print(len(train_paths), len(val_paths))

    train_dl = make_dataLoader(batch_size=16,img_path=train_paths,split="train")
    val_dl = make_dataLoader(batch_size=16,img_path=val_paths,split="val",shuffle=False)