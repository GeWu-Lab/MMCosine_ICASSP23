import argparse
from operator import mod
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.vox.basic_model import AVClassifier
from dataset.vox2_dataset import train_dataset_loader, test_dataset_loader, vox1_test_dataset_loader
from utils.utils import setup_seed, weight_init,re_init
from torch.cuda.amp import autocast, GradScaler


scaler = GradScaler()

def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='vox2', type=str,
                        help='vox2')
    parser.add_argument('--fusion_method', default='film', type=str,
                        choices=['concat', 'gated', 'film'])
    parser.add_argument('--mmcosine',default=False,type=bool,help='whether to involve mmcosine')
    parser.add_argument('--scaling',default=10,type=float,help='scaling parameter in mmCosine')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=200, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')


    #parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--ckpt_path', default='/home/ruize_xu/ruoxuan/CD/log_vox2_cos_new', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=True, type=bool, help='whether to visualize')
    #parser.add_argument('--tensorboard_path', required=True, type=str, help='path to save tensorboard logs')
    parser.add_argument('--tensorboard_path', default='/home/ruize_xu/ruoxuan/CD/log_vox2_cos_new', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0,1', type=str, help='GPU ids')
    parser.add_argument("--save_feature",default=0,type=int,help="option of whether saving features for tsne or not")

    
    
    ## 单模态预训练策略
    parser.add_argument('--audio_pretrain',default='None',type=str,help='path of pretrained audio resnet')
    parser.add_argument('--visual_pretrain',default='None',type=str,help='path of pretrained visual resnet')

       
    return parser.parse_args()

def compute_eer(scores, labels, eps=1e-6):

	# Get the index list after sorting the scores list
	sorted_index = [index for index, value in sorted(enumerate(scores), key=lambda x: x[1])]
	# Sort the labels list
	sorted_labels = [labels[i] for i in sorted_index]
	sorted_labels = np.array(sorted_labels)
	
	FN = np.cumsum(sorted_labels == 1) / (sum(sorted_labels == 1) + eps)
	TN = np.cumsum(sorted_labels == 0) / (sum(sorted_labels == 0) + eps)
	FP = 1 - TN
	TP = 1 - FN
	
	FNR = FN / (TP + FN + eps)
	FPR = FP / (TN + FP + eps)
	difs = FNR - FPR
	idx1 = np.where(difs < 0, difs, float('-inf')).argmax(axis=0)
	idx2 = np.where(difs >= 0, difs, float('inf')).argmin(axis=0)
	x = [FPR[idx1], FPR[idx2]]
	y = [FNR[idx1], FNR[idx2]]

	a = (x[0] - x[1]) / (y[1] - x[1] - y[0] + x[0])
	eer = 100 * (x[0] + a * (y[0] - x[0]))
	
	Cmiss = 1
	Cfa = 1
	P_tgt = 0.01
	
	Cdet = Cmiss * FNR * P_tgt + Cfa * FPR * (1 - P_tgt)
	dcf_voxceleb = 100 * min(Cdet)
	
	
	return eer, dcf_voxceleb


def train_epoch(args, epoch, model, device, dataloader, optimizer, writer=None):
    
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0

    for step, (spec, image, label) in enumerate(dataloader):
        optimizer.zero_grad()
        
        with autocast():
            spec = spec.to(device,non_blocking=True)
            image = image.to(device,non_blocking=True)
            label = label.to(device,non_blocking=True)
            a, v, out = model(spec.squeeze(1).float(), image.float(),args=args)
        

            # TODO: make it simpler and easier to extend
            
            
            
            ## our modality-wise normalization on weight and feature
        if args.mmcosine:
            out_a = torch.mm(F.normalize(a,dim=1), F.normalize(torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1),dim=0))   # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
            out_v = torch.mm(F.normalize(v,dim=1), F.normalize(torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1),dim=0))  
            out_a=out_a*args.scaling
            out_v=out_v*args.scaling
            out=out_a+out_v
        else:
            out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1)) +
                     model.module.fusion_module.fc_out.bias / 2)
            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
                     model.module.fusion_module.fc_out.bias / 2)


            # out = out_a + out_v
            loss = criterion(out, label)
            loss_v = criterion(out_v, label)
            loss_a = criterion(out_a, label)
            loss.backward()
        
            optimizer.step()

        

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        

        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader)


def valid(args, model, device, dataloader,epoch):
    cos = torch.nn.CosineSimilarity(dim=1)
    targets = torch.tensor([]).to(device)
    preds = torch.tensor([]).to(device)

    with torch.no_grad():
        model.eval()

        

        for step, ((spec1,image1), (spec2,image2), label) in enumerate(dataloader):

            
                 
            spec1 = spec1.to(device)
            image1 = image1.to(device)
            spec2 = spec2.to(device)
            image2 = image2.to(device)
            label = label.to(device)
            a1, v1, out1 = model(spec1.squeeze(1).float(), image1.float(),args=args)
            a2, v2, out2 = model(spec2.squeeze(1).float(), image2.float(),args=args)
           

            
            if args.mmcosine:

                out_a1 = torch.mm(F.normalize(a1,dim=1), F.normalize(torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1),dim=0))
                out_v1 = torch.mm(F.normalize(v1,dim=1), F.normalize(torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1),dim=0)) 

                out_a2 = torch.mm(F.normalize(a2,dim=1), F.normalize(torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1),dim=0))
                out_v2 = torch.mm(F.normalize(v2,dim=1), F.normalize(torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1),dim=0)) 

            

                out1=(out_a1+out_v1)*args.scaling
                out2=(out_a2+out_v2)*args.scaling
                # print(out,out_a,out_v)
            else:
                out_a1 = (torch.mm(a1, torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1)) +
                         model.module.fusion_module.fc_out.bias / 2)
                out_v1 = (torch.mm(v1, torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
                         model.module.fusion_module.fc_out.bias / 2)

                out_a2 = (torch.mm(a2, torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1)) +
                         model.module.fusion_module.fc_out.bias / 2)
                out_v2 = (torch.mm(v2, torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
                         model.module.fusion_module.fc_out.bias / 2)
                out1=out_a1+out_v1
                out2=out_a2+out_v2

            

            

            prediction = cos(out1,out2)
            preds = torch.cat((preds,prediction),0)
            targets = torch.cat((targets,label),0)

            # print(prediction)

        
    preds = preds.cpu()
    targets = targets.cpu()
    result = compute_eer(preds,targets)
    return result[0],result[1]

def main():
    args = get_arguments()
    print(args)

    setup_seed(args.random_seed)
    
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')
    model = AVClassifier(args)
    model.to(device)

    

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda()

    # for name,param in model.named_parameters():
    #     print(str(name))

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

    train_dataset = train_dataset_loader(200)
    test_dataset = test_dataset_loader(400,1)
    test_dataset_vox1 = vox1_test_dataset_loader(400,1)
            
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=16,pin_memory=True)
                                  
                             
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=16)

    test_dataloader_vox1 = DataLoader(test_dataset_vox1, batch_size=args.batch_size,
                                 shuffle=False, num_workers=16)

    # err,min_dcf = valid(args, model, device, test_dataloader,1)
    # print(err,min_dcf)
    # exit(0)

    if args.train:
       
        best_err = 100.0
        best_dcf = 100.0
        vox1_best_err = 100.0
        vox1_best_dcf = 100.0

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            if args.use_tensorboard:

                writer_path = os.path.join(args.tensorboard_path)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_optimizer_{}_dataset_{}_mmcosine{}{}'.format(
                                                                                  args.fusion_method,
                                                                                  args.optimizer,                                                                              
                                                                                  args.dataset,
                                                                                  args.mmcosine,
                                                                                  args.scaling
                                                                                                                                                             
                                                                                 )
                writer = SummaryWriter(os.path.join(writer_path, log_name))

                batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, writer)
                scheduler.step()

                err,min_dcf = valid(args, model, device, test_dataloader,epoch)
                err_vox1,min_dcf_vox1 = valid(args, model, device, test_dataloader_vox1,epoch)


                writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                            'Audio Loss': batch_loss_a,
                                            'Visual Loss': batch_loss_v}, epoch)

                writer.add_scalars('Evaluation', {'ERR': err,
                                                  'minDCF': min_dcf,'vox1_ERR': err_vox1,
                                                  'vox1_minDCF': min_dcf_vox1}, epoch)

            else:
                batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer)
                scheduler.step()
                err,min_dcf = valid(args, model, device, test_dataloader,epoch)
                err_vox1,min_dcf_vox1 = valid(args, model, device, test_dataloader_vox1,epoch)

            if err_vox1 < vox1_best_err:
                vox1_best_err = float(err_vox1)

            if min_dcf_vox1 < vox1_best_dcf:
                vox1_best_dcf = float(min_dcf_vox1)

            if err < best_err:
                best_err = float(err)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = 'best_err_model_of_vox2_{}' \
                             'optimizer_{}_mmcosine_{}{}.pth'.format(
                                                                                  args.fusion_method,
                                                                                  args.optimizer,                                                                                  
                                                                                  args.mmcosine,
                                                                                  args.scaling
                                                                                                                                                             
                                                                                  )
                saved_dict = {'saved_epoch': epoch,
                              'scaling': args.scaling,
                              'fusion': args.fusion_method,
                              'err': err,
                              'mindcf':min_dcf,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}
                
                
                save_dir = os.path.join(args.ckpt_path, model_name)
                torch.save(saved_dict, save_dir)
            
            if min_dcf < best_dcf:
                best_dcf = float(min_dcf)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = 'best_dcf_model_of_vox2_{}' \
                             'optimizer_{}_mmcosine_{}{}.pth'.format(
                                                                                  args.fusion_method,
                                                                                  args.optimizer,                                                                                  
                                                                                  args.mmcosine,
                                                                                  args.scaling
                                                                                                                                                             
                                                                                  )
                saved_dict = {'saved_epoch': epoch,
                              'scaling': args.scaling,
                              'fusion': args.fusion_method,
                              'err': err,
                              'mindcf':min_dcf,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}
             
                save_dir = os.path.join(args.ckpt_path, model_name)
                torch.save(saved_dict, save_dir)
            
            
            print("Loss: {:.4f}, Err: {:.2f}, Best Err: {:.2f},  minDCF:{:.3f}, Best minDCF:{:.3f}, vox1 Err: {:.2f}, vox1 Best Err: {:.2f},  vox1 minDCF:{:.3f}, vox1 Best minDCF:{:.3f}".format(batch_loss, err, best_err,min_dcf,best_dcf, err_vox1, vox1_best_err, min_dcf_vox1, vox1_best_dcf))

            

    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model.load_state_dict(state_dict)
        print('Trained model loaded!')

        err,min_dcf = valid(args, model, device, test_dataloader,epoch=1001)
        print('test set:'
			      '\n EER:%.2f  minDCF:%.3f' % (err, min_dcf))




if __name__ == "__main__":
    main()