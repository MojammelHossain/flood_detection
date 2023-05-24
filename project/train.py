import tqdm
import  torch
from utils import *
from model import UNET
from dataset import *
from metric import MeanIoU
USE_AMP = True

def train(dataloader, model, config):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), 0.001, weight_decay=0.0001)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    loss = torch.nn.BCEWithLogitsLoss()
    miou = MeanIoU()
    t = tqdm.tqdm(range(0, config['epochs']))

    for epoch in t:
        tot_loss = 0
        mean_iou = 0
        for i, batch in enumerate(dataloader):
            label = torch.permute(batch[1], (0,3,1,2)).to('cuda')
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                out = model(torch.permute(batch[0], (0,3,1,2)).to('cuda'))
                batch_loss = loss(out, label)
            
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            mean_iou += miou(out, label).item()
            tot_loss += batch_loss.item()
        t.set_description("TRAIN LOSS {}, MEANIOU {}".format(tot_loss/len(dataloader), mean_iou/len(dataloader)))
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, (config['checkpoint_dir']+config['checkpoint_name']))

if __name__ == '__main__':
    config = get_config_yaml("D:/MsCourse/AI/project/flood_detection/project/config.yaml", {})
    create_paths(config)
    dataloader = get_train_val_dataloader(config)
    model = UNET([3,16,32,64,128], [256,128,64,32,16], 2)
    model = model.to(device="cuda", memory_format=torch.channels_last)
    train(dataloader, model, config)