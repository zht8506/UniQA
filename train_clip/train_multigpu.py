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
from utils import convert_models_to_fp32,create_logits

class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,text):
        return self.model.encode_text(text)
    
class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model
        
    def forward(self,image):
        return self.model.encode_image(image)

def main():
    save_all=False # save more in checkpoint, use for resume
    EPOCH=5
    BATCH_SIZE=2
    pretrained_weight="/xxx/pretrain_weight/CLIP-RN50.pt"
    save_path="model_checkpoint_log/"+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+"/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    device_ids = [0, 1, 2, 3]
    model, preprocess = clip.load(pretrained_weight,device=device,jit=False) # Must set jit=False for training

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)

    model_text = torch.nn.DataParallel(model_text,device_ids=device_ids)
    model_image = torch.nn.DataParallel(model_image,device_ids=device_ids)

    # construct dataset
    data_root = ""
    img_cap_path = ["UniQA_dataset.json"]
    print(img_cap_path)
    dataset = image_caption_dataset(data_root,img_cap_path,preprocess)
    train_dataloader = DataLoader(dataset, batch_size = BATCH_SIZE)

    print("dataset length: %d"% len(dataset))
    print("iters of a epoch: %d" % (len(dataset) // BATCH_SIZE))
    print("total iters: %d" % (EPOCH * len(dataset)//BATCH_SIZE))

    with open(save_path+"loss.txt","a") as f:
        f.write("dataset length: %d \n" % len(dataset))
        f.write("iters of a epoch: %d \n"% (len(dataset)//BATCH_SIZE))
        f.write("total iters: %d \n"% (EPOCH * len(dataset)//BATCH_SIZE))
        for img_path_one in img_cap_path:
            f.write("img path: %s \n" % (img_path_one))
        f.write("pretrained weight: %s \n" % (pretrained_weight))

    if device == "cpu":
        model.float()
    else :
        clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    # loss and optimizer
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    lr = 5e-6
    optimizer = optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    # train loop
    for epoch in range(1,EPOCH+1):
        for i,batch in enumerate(train_dataloader):
            if batch[1].shape[0] != BATCH_SIZE:
                continue
            optimizer.zero_grad()

            images,texts = batch 
            images= images.to(device)
            texts = texts.to(device)

            image_embedding = model_image(images)
            text_embedding = model_text(texts)
            
            logit_scale = model.logit_scale.exp()
            logits_per_image, logits_per_text = create_logits(image_embedding,text_embedding,logit_scale)
            import pdb; pdb.set_trace()
            ground_truth = torch.arange(BATCH_SIZE).to(device)
            

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

        if True:
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