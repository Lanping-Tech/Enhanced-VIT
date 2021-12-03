import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random

import torch
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import mse_loss, l1_loss
from torchvision import transforms
import numpy
import torch.backends.cudnn as cudnn
from sklearn import metrics
import argparse
from tqdm import tqdm

import datasets



def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Image Nonlinear Regression'
    )

    # dataset
    parser.add_argument('--dataset_name', default='CIFAR10', type=str, help='dataset name')
    parser.add_argument('--train_batch_size', default=100, type=int, help='training batch size')
    parser.add_argument('--test_batch_size', default=100, type=int, help='testing batch size')

    # model
    parser.add_argument('--model_name', default='ResNet18', type=str, help='model name')
    
    parser.add_argument('--backbone', type=str, default='resnet18', help='the name of backbone')

    parser.add_argument('--data_path', type=str, default='dataset', help='the path of dataset')

    parser.add_argument('--epochs', default=50, type=int, help='number of epochs tp train for')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str, help='divice')

    parser.add_argument('--output_path', default="output", type=str, help='output path')
    parser.add_argument('--seed', type=int, default=1, help='Random seed, a int number')
    
    return parser.parse_args()

def test(model, dataset, batch_size, device, test=False):

    model = model.eval()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    predict_all = numpy.array([], dtype=int)
    labels_all = numpy.array([], dtype=int)
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            data, target = batch
            data = data.double().to(device)
            target = target.double().to(device)

            y_pred = model(data)

            target = target.data.cpu().numpy()
            predict = y_pred.data.cpu().numpy()
            labels_all = numpy.append(labels_all, target)
            predict_all = numpy.append(predict_all, predict)

    mae = metrics.mean_absolute_error(labels_all, predict_all)
    mse = metrics.mean_squared_error(labels_all, predict_all)
    r2 = metrics.r2_score(labels_all, predict_all)

    if test:
        return labels_all, predict_all, mae, mse, r2
    else:
        return mae, mse, r2

def train(model, train_dataset, test_dataset, args):

    train_data_loader = torch.utils.data.DataLoader(train_dataset, args.train_batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    train_maes, train_mses, train_r2s, test_maes, test_mses, test_r2s = [],[],[],[],[],[]

    for epoch in range(args.epochs):
        model = model.train()
        pbar = tqdm(train_data_loader)
        pbar.set_description("Epoch {}:".format(epoch))
        for data, target in pbar:

            data = data.double().to(args.device)
            target = target.double().to(args.device)

            optimizer.zero_grad()
            predict = model(data)
            loss = mse_loss(predict, target)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            pbar.set_postfix(loss=loss.item())
        
        # val phase
        train_mae, train_mse, train_r2 = test(model, train_dataset, args.train_batch_size, args.device)
        test_mae, test_mse, test_r2 = test(model, test_dataset, args.test_batch_size, args.device)
        train_maes.append(train_mae)
        train_mses.append(train_mse)
        train_r2s.append(train_r2)
        test_maes.append(test_mae)
        test_mses.append(test_mse)
        test_r2s.append(test_r2)
        
        # model save
        torch.save(model.state_dict(), args.output_path+'/epoch_{0}_train_loss_{1:>0.5}_test_loss_{2:>0.5}.ckpt'.format(epoch,train_mse,test_mse))

    
    mae_plot = {}
    mae_plot['train'] = train_maes
    mae_plot['test'] = test_maes

    mse_plot = {}
    mse_plot['train'] = train_mses
    mse_plot['test'] = test_mses

    r2_plot = {}
    r2_plot['train'] = train_r2s
    r2_plot['test'] = test_r2s

    train_labels_all, train_predict_all, _, _, _ = test(model, train_dataset, args.train_batch_size, args.device)
    train_value_plot = {}
    train_value_plot['label'] = train_labels_all
    train_value_plot['predict'] = train_predict_all

    test_labels_all, test_predict_all, _, _, _ = test(model, test_dataset, args.test_batch_size, args.device)
    test_value_plot = {}
    test_value_plot['label'] = test_labels_all
    test_value_plot['predict'] = test_predict_all

    performance_display(mae_plot, 'MAE', args.output_path)
    performance_display(mse_plot, 'MSE', args.output_path)
    performance_display(r2_plot, 'R2', args.output_path)
    performance_display(train_value_plot, 'Train Regression', args.output_path)
    performance_display(test_value_plot, 'Test Regression', args.output_path)

if __name__ == '__main__':
    args = parse_arguments()
    seed_torch(args.seed)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load train and test data
    train_loader, test_loader = getattr(datasets, 'load_'+args.dataset_name)(args.train_batch_size, args.test_batch_size)

    image_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()])

    train_data = ImageCSVData(image_train_path, label_train_path, image_transforms)
    test_data = ImageCSVData(image_test_path, label_test_path, image_transforms)

    # build model
    model = Model(args.backbone)
    model = model.double().to(args.device)

    # train phase
    train(model, train_data, test_data, args)