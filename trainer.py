from model.layers.metrics import MetricHolder
from model.model import *
import torch
from utils.fenchel_young_loss import * 
from dba.data_dealer import * 
from model.layers.differential_ranking import * 
from pathlib import Path
from utils.saver import * 
from utils.config import * 
from tqdm import tqdm
from utils.log import logger
from utils.mask import gen_sentence_mask
import time

def test(args,  checkpoint):
    device=DEVICE
    model = Network(args.d_mlp)
    model=model.to(device)
    checkpoint = torch.load("/content/checkpoints_pytorch/model_0")
    model.load_state_dict(checkpoint['model_state_dict'])
     
def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters()]))
 

def gen_model_and_optimizer(args,device,device2):
    model = Network(device,device2,args.parallel,args.bert_type,args.predict_type,args.global_encoder)
    if args.parallel =='none':
        model=model.to(device)
    print_params(model)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
    if args.amp=='Y':
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    return model,optimizer

def train(args):
    start = time.time()
    train_dataloader,val_dataloader=load_data(args)
    # device=DEVICE
    if args.parallel =='model':
        device= torch.device('cuda:'+args.gpu )#
        device2=torch.device('cuda:'+args.gpu2)#
    else:
        device = DEVICE
        device2= DEVICE
    print(f"use {device}")
    model,optimizer=gen_model_and_optimizer(args,device,device2)
    criterion = FenchelYoungLoss()
    valid_metric_holder=MetricHolder(args.metric) 
    train_metric_holder=MetricHolder(args.metric) 
     
    for epoch in range(args.max_epochs):
        train_loss,train_score_str=train_one_epoch(train_dataloader,device,model,criterion,optimizer,args,epoch,device2,train_metric_holder)

        # DO EVAL ON VALIDATION SET
        val_loss,best_score,best_epoch=validate(model,val_dataloader,device,criterion,epoch,train_loss,train_score_str,args,valid_metric_holder)
        
         
        save_best_model(epoch,best_score,best_epoch, model,args,optimizer,criterion)

    
        if args.early_stop and (epoch - best_epoch) >= args.early_stop:
            print('early stop at epc {} with best_score{}'.format(epoch,best_score))
            break
     
    minutes = (time.time() - start) // 60
    hours = minutes / 60
    print('use time:{:.1f} hours '.format( hours ))    
    

def train_one_epoch(train_dataloader,device,model,criterion,optimizer,args,epoch,device2,metric_holder):
    
    metric_holder.epoch_reset()
    over_loss =0
    train_steps = 0 
    model.train()

    if epoch<1:
        lr = 5e-7
    else:
        lr = 1e-7
    for g in optimizer.param_groups:
        g['lr'] = lr
    print("Learning Rate is: \n", lr)
    for steps, (input_ids, attn_masks,sentence_num_list, train_labels) in enumerate(tqdm(train_dataloader)):
        
        train_labels = train_labels.to(device)


        out,loss = model(input_ids, attn_masks,sentence_num_list,train_labels)


        if args.criterion=="fenchel_young":  #TODO 
            masked_out=mask_out(out ,sentence_num_list,train_labels.float())
            loss = criterion(masked_out , train_labels.float()).mean()

        # Backward and optimize
        optimizer.zero_grad()
        if args.amp=='Y':
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # compute accuracy
        over_loss += loss
        metric_holder.compute( out,train_labels,sentence_num_list ,args.criterion )
        
        if steps%1000==0:
            score_str=metric_holder.avg_step_score_str(steps+1)
            print(('Step: %1.1f, Loss: % 2.5f,  % s ' %(steps,over_loss/(steps+1),score_str)))
        train_steps = steps+1
        
    return over_loss/train_steps,metric_holder.avg_step_score_str(train_steps)





def mask_out(out,sentence_num_list,train_labels):
    mask=gen_sentence_mask(out,sentence_num_list)
    mask_out=torch.where(mask,out,train_labels)
    return mask_out


def save_best_model(epoch,best_score,best_epoch,model,args,optimizer,criterion):
     
    if epoch==best_epoch:
        main_path = Path(args.output_parent_dir)
        model_path, log_path=get_output_path_str(main_path)
        PATH = model_path+'/model_'+args.time_flag

        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "args":args,
                'loss': criterion,
                "score":best_score
                }, PATH)
        logger.info(f"save best model at epoch {epoch} with score {best_score}")
    return

def validate(model,val_dataloader,device,criterion,epoch,train_loss,train_score_str,args,metric_holder ):
    metric_holder.epoch_reset()
    val_loss = 0
     
    val_steps = 0
    model.eval()
    for steps, (input_ids, attn_masks,sentence_num_list, val_labels) in enumerate(tqdm(val_dataloader)):

    
        val_labels = val_labels.to(device)

        with torch.no_grad():

            out,loss = model(input_ids, attn_masks,sentence_num_list,val_labels) 

            if args.criterion=="fenchel_young":     
                masked_out=mask_out(out,sentence_num_list,val_labels.float())      
                loss = criterion(masked_out , val_labels.float()).mean()
            val_loss += loss
            metric_holder.compute( out,val_labels,sentence_num_list ,args.criterion)
   
        
        val_steps = steps+1
    best_score,best_epoch,val_score_str=metric_holder.update_epoch_score(epoch,val_steps )
    c = (('Epoch: %1.1f, Training Loss: % 2.5f, Training % s , Validation Loss: % 4.5f, Validation % s  ' %(epoch,train_loss,train_score_str, val_loss/val_steps,val_score_str)))
    print(c)       
    logger.info(c)
    return val_loss/val_steps,best_score,best_epoch

            
         