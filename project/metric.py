import torch
from model import UNET
from dataset import get_train_val_dataloader
from utils import get_config_yaml
import torch
from segmentation_models_pytorch.metrics import iou_score


class MeanIoU:
    def __init__(self):
        self.epsilon = 1e-10

    def __call__(self, tensor1, tensor2):
        # if single dimension
        if len(tensor1.shape) == 1 and len(tensor2.shape) == 1:
            inter = torch.sum(torch.squeeze(tensor1 * tensor2))
            union = torch.sum(torch.squeeze(tensor1 + tensor2)) - inter
        else:
            inter = torch.sum(
                torch.sum(torch.squeeze(tensor1 * tensor2, axis=3), axis=2), axis=1
            )
            union = (
                torch.sum(
                    torch.sum(torch.squeeze(tensor1 + tensor2, axis=3), axis=2), axis=1
                )
                - inter
            )
        return torch.mean((inter + self.epsilon) / (union + self.epsilon))

if __name__ == '__main__':
    config = get_config_yaml("D:/MsCourse/AI/project/flood_detection/project/config.yaml", {})
    dataloader = get_train_val_dataloader(config)
    model = UNET([3,16,32,64,128], [256,128,64,32,16], 2).to("cuda")
    checkpoint = torch.load("D:/MsCourse/AI/project/flood_detection/model/unet/unet_ex_patchify_epochs_2000_23-May-23.pt")
    for i, batch in enumerate(dataloader):
        out = out = model(torch.permute(batch[0], (0,3,1,2)).to('cuda'))
        break
    m = MeanIoU()
    print(m(out, torch.permute(batch[1], (0,3,1,2)).to('cuda')).item())