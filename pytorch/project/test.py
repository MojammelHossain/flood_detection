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
    b = 100
    with torch.no_grad():
        for i, (feature, mask) in enumerate(dataloader):
            label = torch.permute(mask, (0,3,1,2)).to('cuda')
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                out = model(torch.permute(feature, (0,3,1,2)).to('cuda'))
                batch_loss = loss(out, label)
            tot_loss += batch_loss.item()
            print("Batch : ", batch_loss.item())
            mean_iou += miou(out, label).item()
            if b==i:
                break

    tot_loss = tot_loss / len(dataloader)
    mean_iou = mean_iou / len(dataloader)
    if val:
        model.train()
    return tot_loss, mean_iou

if __name__ == '__main__':
    config = get_config_yaml("D:/MsCourse/AI/project/flood_detection/project/config.yaml", {})
    create_paths(config, True)
    dataloader = get_test_dataloader(config)
    model = UNET([3,16,32,64,128], [256,128,64,32,16], 2)
    model = model.to(device="cuda", memory_format=torch.channels_last)
    print("Loading model from checkpoint : {}".format(config['load_model_name']))
    print((config['load_model_dir']+config['load_model_name']))
    checkpoint = torch.load((config['load_model_dir']+config['load_model_name']))
    model.load_state_dict(checkpoint["model"])
    show_predictions(dataloader, model, config)
    loss, iou = evaluate(dataloader, model, True)
    print("TEST LOSS {:.4f}, TEST MEANIOU {:.4f}".format(loss, iou))