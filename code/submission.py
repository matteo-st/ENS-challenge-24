from models import UNet
import os 

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from einops import rearrange

from dataset.chd import  RaidiumUnlabeled, raidius_sup_collate
from myconfig import get_config
from network.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from network.vit_seg_modeling import VisionTransformer as ViT_seg

# Segmentation
from meanshift import MeanShiftCluster

def load_model(args):
    config_vit = CONFIGS_ViT_seg[args["vit_name"]]
    config_vit.n_classes = args["classes"]
    config_vit.n_skip = args["n_skip"]
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    model = ViT_seg(config_vit, img_size=args["img_size"], num_classes=args["classes"]).cuda()
    print(os.getcwd())
    # model.load_from(weights=np.load(config_vit.pretrained_path))
    # model = UNet(args["input_dim"], args["classes"])
    model.load_state_dict(
        torch.load(args["model_path"])["model_state_dic"]
        )
    return model


def create_submission_file(model, data_loader, args):
    args["device"] = torch.device(
         args["device"] if torch.cuda.is_available() else "cpu")
    print("The model will run on device:", args["device"])
    
    # Tensor type (put everything on GPU if possible)
    

    model.to(args["device"])

    meanshift = MeanShiftCluster()
    submissions = pd.DataFrame(columns=[f'Pixel {i}' for i in range(512*512)])

    with torch.no_grad():
        for sample in tqdm(data_loader):
            img = sample["slice"].float().to(args["device"])
            name_file = sample['image_file'][0]

            # Creating prediction for unlabeled data
            label_logits = model(img)
            seg = meanshift(label_logits).cpu().numpy().flatten().astype(np.uint8)
            submissions.loc[name_file] = seg.tolist()
    
    submission = submissions.transpose()
    print("Submission shape : ", submission.shape)
    
    submissions.transpose().to_csv(
        os.path.join(args["submit_path"], "y_submit.csv")
        )


args = {
    "classes" : 1,
    "input_dim" : 1,
    "model_path" : '../checkpoints/sup/_TransUNetbs-4_2024-03-16_20-32-21/epoch-98.pth',
    "submit_path" : '../submissions',
    "device" : 'cuda:0',
    "vit_name" : "ViT-B_16",
    "n_skip" : 0,
    "img_size" : 512
}

if __name__ == '__main__':

    print("Submission...")
    
    # Creating submission directory
    model_name = args["model_path"].split("/")[-2]
    args["submit_path"] = os.path.join(args["submit_path"], model_name)
    if not os.path.exists(args["submit_path"]):
        os.makedirs(args["submit_path"])

    # Loading trained model
    model = load_model(args)
    print("Model {model} imported.".format(model=model_name))
    test_dataset = RaidiumUnlabeled(path="../data/x_test",
                            args=None)
    test_loader = torch.utils.data.DataLoader(
         test_dataset, batch_size=1, shuffle=False,
        # num_workers=args.num_works, 
        drop_last=False, collate_fn=raidius_sup_collate
        )
    print(f"Data imported. Nb files : {len(test_dataset)}")
    

    create_submission_file(model, test_loader, args)

    print("Submission created.")



