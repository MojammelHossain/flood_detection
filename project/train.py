import tqdm
import  torch
from utils import *
from model import UNET
from dataset import *
from metric import MeanIoU
USE_AMP = False

def evaluate(dataloader, model, val=False):
    model.eval()
    loss = torch.nn.BCELoss()
    miou = MeanIoU()
    tot_loss = 0
    mean_iou = 0
    b = 10
    with torch.no_grad():
        for i, (feature, mask) in enumerate(dataloader):
            label = torch.permute(mask, (0,3,1,2)).to('cuda')
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                out = model(torch.permute(feature, (0,3,1,2)).to('cuda'))
                batch_loss = loss(out, label)
            tot_loss += batch_loss.item()
            mean_iou += miou(out, label).item()
            if i==b:
                break
    tot_loss = tot_loss / len(dataloader)
    mean_iou = mean_iou / len(dataloader)
    if val:
        model.train()
    return tot_loss, mean_iou


def train(dataloader, val_dataloader, model, config):
    optimizer = torch.optim.AdamW(model.parameters(), 0.001, weight_decay=0.0001)
    #lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, end_factor=0.5, total_iters=1000)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    start_epoch = 0

    if config['load_model_name'] != "None":
        print("Loading model from checkpoint : {}".format(config['load_model_name']))
        checkpoint = torch.load((config['load_model_dir']+config['load_model_name']))
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    model.train()
    loss = torch.nn.BCELoss()
    miou = MeanIoU()
    t = tqdm.tqdm(range(start_epoch, config['epochs']))

    for epoch in t:
        tot_loss = 0
        mean_iou = 0
        b = 10
        for i, (feature, mask) in enumerate(dataloader):
            label = torch.permute(mask, (0,3,1,2)).to('cuda')
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                out = model(torch.permute(feature, (0,3,1,2)).to('cuda'))
                batch_loss = loss(out, label)
            
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # lr_scheduler.step()
            # print("LEARNING RATE {}".format(lr_scheduler.get_lr()))
            mean_iou += miou(out, label).item()
            tot_loss += batch_loss.item()
            if b==i:
                break
        if epoch % config['val_plot_epoch'] == 0:
            show_predictions(val_dataloader, model, config, True)
        val_loss, val_iou = evaluate(val_dataloader, model, True)
        t.set_description("TRAIN LOSS {:.4f}, TRAIN MEANIOU {:.4f}, VAL LOSS {:.4f}, VAL MEANIOU {:.4f}"
                          .format(tot_loss/len(dataloader), mean_iou/len(dataloader),
                                  val_loss, val_iou))
        write_csv(config, {"epoch": epoch,
                   "train_loss": tot_loss/len(dataloader),
                   "train_mean_iou": mean_iou/len(dataloader),
                   "val_loss": val_loss,
                   "val_mean_iou": val_iou})
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            # 'lr_scheduler': lr_scheduler.state_dict()
        }, (config['checkpoint_dir']+config['checkpoint_name']))

if __name__ == '__main__':
    config = get_config_yaml("D:/MsCourse/AI/project/flood_detection/project/config.yaml", {})
    create_paths(config)
    dataloader, val_dataloader = get_train_val_dataloader(config)
    model = UNET([3,16,32,64,128], [256,128,64,32,16], 2)
    model = model.to(device="cuda", memory_format=torch.channels_last)
    train(dataloader, val_dataloader, model, config)
