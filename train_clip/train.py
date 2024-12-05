import torch
import clip
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torch.nn as nn
import os
import pandas as pd
import torch.optim as optim
from dataset import image_caption_dataset
import datetime
from utils import convert_models_to_fp32

def main():
    save_all=True # save more in checkpoint, use for resume
    EPOCH=50
    BATCH_SIZE=196
    pretrained_weight="/data/zht/pretrain_weight/CLIP-RN50.pt"
    save_path="model_checkpoint_log/"+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+"/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model, preprocess = clip.load(pretrained_weight,device=device,jit=False) # Must set jit=False for training

    # construct dataset
    data_root = "/data/iaa_data/AVA_dataset/image"
    img_cap_path = "/data/iaa_data/ava_captions.json"
    dataset = image_caption_dataset(data_root,img_cap_path,preprocess)
    train_dataloader = DataLoader(dataset, batch_size = BATCH_SIZE)

    print("dataset length: %d"% len(dataset))
    print("iters of a epoch: %d" % (len(dataset) // BATCH_SIZE))
    print("total iters: %d" % (EPOCH * len(dataset)//BATCH_SIZE))

    with open(save_path+"loss.txt","a") as f:
        f.write("dataset length: %d" % len(dataset))
        f.write("iters of a epoch: %d"% (len(dataset)//BATCH_SIZE))
        f.write("total iters: %d"% (EPOCH * len(dataset)//BATCH_SIZE))

    if device == "cpu":
        model.float()
    else :
        clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    # loss and optimizer
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    # train loop
    for epoch in range(1,EPOCH+1):
        for i,batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            images,texts = batch 
            images= images.to(device)
            texts = texts.to(device)
            
            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            total_loss.backward()
            print("%d %d loss: %.4f"%(epoch,i,total_loss.detach().cpu().numpy()))
            if device == "cpu":
                optimizer.step()
            else : 
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
            with open(save_path+"loss.txt","a") as f:
                f.write("%d %d loss: %.4f \n"%(epoch,i,total_loss.detach().cpu().numpy()))

        if epoch%5==0 or epoch==1:
            if save_all and epoch%5==0: # save_all for resume
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                    }, save_path+"model_"+str(epoch)+".pt")
            torch.save(model.state_dict(),
                    save_path+"model_state"+str(epoch)+".pt")

if __name__=="__main__":
    main()