import argparse
from operator import mod
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from dataset.dataset import AVDataset_CD
from models.basic_model import AVClassifier
from utils.utils import setup_seed, weight_init,re_init

EPISILON=1e-10
# NCE loss is used in supplementary material
class NCELoss(torch.nn.Module):

  def __init__(self, temperature=0.1):
    super(NCELoss, self).__init__()
    self.temperature = temperature
    self.softmax = nn.Softmax(dim=1)

  def where(self, cond, x_1, x_2):
    cond = cond.type(torch.float32)
    return (cond * x_1) + ((1 - cond) * x_2)

  def forward(self, f1, f2, targets):
    ### cuda implementation
    f1 = F.normalize(f1, dim=1)
    f2 = F.normalize(f2, dim=1)

    ## set distances of the same label to zeros
    mask = targets.unsqueeze(1) - targets
    self_mask = (torch.zeros_like(mask) != mask).float()  ### where the negative samples are labeled as 1
    dist = (f1.unsqueeze(1) - f2).pow(2).sum(2)

    ## convert l2 distance to cos distance
    cos = 1 - 0.5 * dist

    ## convert cos distance to exponential space
    pred_softmax = self.softmax(cos / self.temperature) ### convert to multi-class prediction scores

    log_pos_softmax = - torch.log(pred_softmax + EPISILON) * (1 - self_mask.float())
    log_neg_softmax = - torch.log(1 - pred_softmax + EPISILON) * self_mask.float()
    log_softmax = log_pos_softmax.sum(1) / (1 - self_mask).sum(1).float() + log_neg_softmax.sum(1) / self_mask.sum(1).float()
    loss = log_softmax

    return loss.mean()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, defult="CREMAD",type=str,
                        help='CREMAD')
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['concat', 'gated', 'film'])
    parser.add_argument('--mmcosine',default=False,type=bool,help='whether to involve mmcosine')
    parser.add_argument('--scaling',default=10,type=float,help='scaling parameter in mmCosine')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=100, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--alpha', default=0, type=float, help='alpha in OGM-GE')

    #parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--ckpt_path', default='./log', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=True, type=bool, help='whether to visualize')
    #parser.add_argument('--tensorboard_path', required=True, type=str, help='path to save tensorboard logs')
    parser.add_argument('--tensorboard_path', default='/home/ruize_xu/CD/log_normal', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0, 1', type=str, help='GPU ids')
    parser.add_argument("--lam",type=float,default=0)
    
    
    ## uni-modal pretrained checkpoint(mainly for ssw)
    parser.add_argument('--audio_pretrain',default='None',type=str,help='path of pretrained audio resnet')
    parser.add_argument('--visual_pretrain',default='None',type=str,help='path of pretrained visual resnet')

       

    return parser.parse_args()


def train_epoch(args, epoch, model, device, dataloader, optimizer, writer=None):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    NCE=NCELoss()
    model.train()
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0

    for step, (spec, image, label, name) in enumerate(dataloader):
        optimizer.zero_grad()
        
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)
        a, v, out = model(spec.unsqueeze(1).float(), image.float(),args=args)
        nce_loss=NCE(a,v,label)
       
## our modality-wise normalization on weight and feature
        if args.mmcosine:
            out_a = torch.mm(F.normalize(a,dim=1), F.normalize(torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1),dim=0))   # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
            out_v = torch.mm(F.normalize(v,dim=1), F.normalize(torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1),dim=0))  
            out_a=out_a*args.scaling
            out_v=out_v*args.scaling
        else:
            out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1)) +
                     model.module.fusion_module.fc_out.bias / 2)
            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
                     model.module.fusion_module.fc_out.bias / 2)
                     

       

        if args.use_tensorboard:
                label_onehot=nn.functional.one_hot(label,num_classes=out_v.size(1))
                fy_v=torch.mean(torch.sum(out_v*label_onehot,dim=1))
                fy_a=torch.mean(torch.sum(out_a*label_onehot,dim=1))
                iteration = epoch * len(dataloader) + step
                #print(ma.shape)
                writer.add_scalar('data/logit_a', fy_a, iteration)
                writer.add_scalar('data/logit_v', fy_v, iteration)
            




        loss = criterion(out, label)+args.lam*nce_loss
        loss_v = criterion(out_v, label)
        loss_a = criterion(out_a, label)
        loss.backward()
        
        optimizer.step()

        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader)


def valid(args, model, device, dataloader,epoch):
    softmax = nn.Softmax(dim=1)

    
    if args.dataset == 'CREMAD':
        n_classes = 6
    
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]
        

        for step, (spec, image, label, name) in enumerate(dataloader):

            
                 
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)
            a, v, out = model(spec.unsqueeze(1).float(), image.float(),args=args)
           
            
            # approximate uni-modal evaluation
            if args.mmcosine:
                out_a = torch.mm(F.normalize(a,dim=1), F.normalize(torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1),dim=0))   # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
                out_v = torch.mm(F.normalize(v,dim=1), F.normalize(torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1),dim=0))  
                out_a=out_a*args.scaling
                out_v=out_v*args.scaling
            else:
                out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1)) +
                        model.module.fusion_module.fc_out.bias / 2)
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
                        model.module.fusion_module.fc_out.bias / 2)

            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i, item in enumerate(name):

                ma = prediction[i].cpu().data.numpy()
                index_ma = np.argmax(ma)
                v = pred_v[i].cpu().data.numpy()
                index_v = np.argmax(v)
                a = pred_a[i].cpu().data.numpy()
                index_a = np.argmax(a)
                num[label[i]] += 1.0
                if index_ma == label[i]:
                    acc[label[i]] += 1.0
                if index_v == label[i]:
                    acc_v[label[i]] += 1.0
                if index_a == label[i]:
                    acc_a[label[i]] += 1.0

        
    return sum(acc) / sum(num),  sum(acc_a) / sum(num) , sum(acc_v) / sum(num),


def main():
    args = get_arguments()
    print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')
    model = AVClassifier(args)
    model.to(device)

    

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda()



    if args.audio_pretrain!='None':
            
            loaded_dict_audio = torch.load(args.audio_pretrain)
            
            
            state_dict_audio = loaded_dict_audio
            

            model.module.audio_net.load_state_dict(state_dict_audio,strict=False)

         
    if args.visual_pretrain!='None':
            
            loaded_dict_visual = torch.load(args.visual_pretrain)
            
            state_dict_visual = loaded_dict_visual

            
            

            model.module.visual_net.load_state_dict(state_dict_visual,strict=False)  

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),lr=args.learning_rate,betas=(0.9, 0.999),eps=1e-08,weight_decay=1e-4,amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset=='CREMAD':
        train_dataset = AVDataset_CD(args, mode='train')
        test_dataset = AVDataset_CD(args, mode='test')
   
            

    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=16, pin_memory=True)
                                  
                             
    
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=16)

    if args.train:      
     
        best_acc = 0.0

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            
            if args.use_tensorboard:

                writer_path = os.path.join(args.tensorboard_path)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_alpha_{}_optimizer_{}_pre_a{}_pre_v{}_dataset_{}_mmcosine{}{}'.format(
                                                                                  args.fusion_method,
                                                                                  args.alpha,
                                                                                  args.optimizer,                                                                              
                                                                                  args.audio_pretrain,
                                                                                  args.visual_pretrain,
                                                                                  args.dataset,
                                                                                  args.mmcosine,
                                                                                  args.scaling
                                                                                                                                                             
                                                                                 )
                writer = SummaryWriter(os.path.join(writer_path, log_name))

                batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, writer)
                scheduler.step()
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader,epoch)

                writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                            'Audio Loss': batch_loss_a,
                                            'Visual Loss': batch_loss_v}, epoch)

                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Audio Accuracy': acc_a,
                                                  'Visual Accuracy': acc_v}, epoch)

            else:
                batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer)
                scheduler.step()
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader,epoch)

            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = 'best_model_of_dataset_{}_{}_alpha_{}_' \
                             'optimizer_{}_pre_a{}_pre_v{}_mmcosine_{}{}.pth'.format(args.dataset,
                                                                                  args.fusion_method,
                                                                                  args.alpha,
                                                                                  args.optimizer,                                                                                  
                                                                                  args.audio_pretrain,
                                                                                  args.visual_pretrain,
                                                                                  args.mmcosine,
                                                                                  args.scaling
                                                                                                                                                             
                                                                                  )
                saved_dict = {'saved_epoch': epoch, 
                              'alpha': args.alpha,
                              'fusion': args.fusion_method,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}
                
                
                
                    
                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.4f}, Acc: {:.4f}, Acc_a:{:.4f}, Acc_v:{:.4f}".format(batch_loss, acc,acc_a,acc_v))
            else:
                print("Loss: {:.4f}, Acc: {:.4f}, Best Acc: {:.4f},  Acc_a:{:.4f}, Acc_v:{:.4f}".format(batch_loss, acc, best_acc,acc_a,acc_v))
    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model = model.load_state_dict(state_dict)
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader,epoch=1001)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()