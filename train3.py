import argparse
import pandas as pd
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.optim as optim
import torch
import torch.nn as nn
import random
from network import Net
import torch
import random
import numpy as np
import cv2

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmarking for reproducibility

seed = 42
set_seed(seed)



parser = argparse.ArgumentParser(description='PyTorch Model Training')

parser.add_argument('--name',default='v8', type=str,
                    help='Name of the experiment.')
parser.add_argument('--out_file', default='new_out.txt',
                    help='path to output features file')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch_size', default=8, type=int,#original 256
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--resume',
                    default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data', default='combine_train.csv', metavar='DIR',
                    help='path to imagelist file')
parser.add_argument('--val_data', default='valid2.csv', metavar='DIR',
                    help='path to imagelist file')
parser.add_argument('--print_freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--epochs', default=501, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number')
parser.add_argument('--save_freq', default=5, type=int,
                    help='Number of epochs to save after')
parser.add_argument('--out_npy_file', default='losses.npy',
                    help='Path to save the numpy file containing train and valid losses')


class CustomImages(Dataset):
    def __init__(self, path_to_trainset, transform=None):
        self.dataset = pd.read_csv(path_to_trainset, sep=' ')
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.dataset.iloc[idx, 0].split(',')[0]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        target = self.dataset.iloc[idx, 0].split(',')[5:9]
        
        if self.transform:
            image = np.expand_dims(image, axis=0)  # Add channel dimension
            image = torch.from_numpy(np.array(image, dtype=np.float32))
            target = torch.from_numpy(np.array(np.asarray(target), dtype=np.float32))
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.dataset)


   
def main():
    args = parser.parse_args()
    print(args)

    print("=> creating model")
    model = Net()

    if args.resume:
        print("=> loading checkpoint: " + args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)
        args.start_epoch = int(args.resume.split('/')[1].split('_')[0])
        print("=> checkpoint loaded. epoch : " + str(args.start_epoch))

    else:
        print("=> Start from the scratch ")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)

    #criterion1 = CustomMSELoss(weight_y=2.0, weight_x=1.0)
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), args.lr)

    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    trainset = CustomImages(
        args.data,
        transforms.Compose([
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    validset = CustomImages(
        args.val_data,
        transforms.Compose([
            normalize,
        ]))
    valid_loader = torch.utils.data.DataLoader(
        validset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)
    
    output= open(args.out_file, "w")

    train_losses = []
    valid_losses = []
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train_loss = train(train_loader,valid_loader, model, criterion1, criterion2, optimizer, epoch, args, device, len(trainset), output)
        train_losses.append(train_loss)

        # Validate
        valid_loss = validate(valid_loader, model, criterion1, criterion2, device)
        valid_losses.append(valid_loss)


    np.save(args.out_npy_file, np.array([train_losses, valid_losses]))

def train(train_loader, valid_loader,model, criterion1, criterion2,optimizer, epoch, args, device, len, file):

    # switch to train mode
    model.train()
    running_loss = 0.0

    for i, (images, target) in enumerate(train_loader):

        images = images.to(device)
        target = target.to(device)

        output = model(images)

        loss = criterion1(output, target) + 0.1 * criterion2(output, target)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % args.print_freq == args.print_freq - 1 or i == int(len/args.batch_size):    # print every 50 mini-batches
            new_line = 'Epoch: [%d][%d/%d] Train Loss: %f, Valid Loss: %f' % \
                       (epoch + 1, i + 1, int(len/args.batch_size) + 1, running_loss / args.print_freq, 
                        validate(valid_loader, model, criterion1, criterion2, device))
            file.write(new_line + '\n')
            print(new_line)
            running_loss = 0.0

        if epoch % args.save_freq == 0:
            torch.save(model.module.state_dict(), 'saved_models/' + str(epoch) + '_epoch_' + args.name + '_checkpoint.pth.tar')

def validate(valid_loader, model, criterion1, criterion2, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, target in valid_loader:
            images = images.to(device)
            target = target.to(device)

            # Forward pass
            output = model(images)

            # Compute loss
            loss = criterion1(output, target) + 0.1 * criterion2(output, target)

            val_loss += loss.item()

    return val_loss / len(valid_loader)

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate"""
    lr = args.lr
    if 100 < epoch <= 200:
        lr = 0.0001
    elif 200 < epoch :
        lr = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("learning rate -> {}\n".format(lr))


if __name__ == '__main__':
    main()
