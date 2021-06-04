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
from utils.mask import gen_mask


def test(args,  checkpoint):
    device=DEVICE
    model = Network(args.d_mlp)
    model=model.to(device)
    checkpoint = torch.load("/content/checkpoints_pytorch/model_0")
    model.load_state_dict(checkpoint['model_state_dict'])
     
def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters()]))
 

def gen_model_and_optimizer(args,device,device2):
    model = Network(device,device2,args.parallel,args.d_mlp)
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
    best_score = -np.inf
    best_iter = 0
    for epoch in range(args.max_epochs):
        train_loss,train_acc,train_pmr=train_one_epoch(train_dataloader,device,model,criterion,optimizer,args,epoch,device2)

        # DO EVAL ON VALIDATION SET
        val_loss,val_acc,val_pmr=validate(model,val_dataloader,device,criterion,epoch,train_loss,train_acc,train_pmr)
        
        best_iter,best_score=save_best_model(epoch,val_acc,val_pmr,best_iter,best_score,model,args,optimizer,criterion)

    
        if args.early_stop and (epoch - best_iter) >= args.early_stop:
            print('early stop at epc {}'.format(epoch))
            break
        
    

def train_one_epoch(train_dataloader,device,model,criterion,optimizer,args,epoch,device2):
    over_acc = 0
    over_pmr=0
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


        out = model(input_ids, attn_masks,sentence_num_list)

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
        pmr,acc= my_metric_fn(train_labels, out,sentence_num_list)
        over_acc +=acc
        over_pmr+=pmr
        if steps%1000==0:
            print(('Step: %1.1f, Loss: % 2.5f, Accuracy % 3.4f%%, Pmr % 3.4f%%' %(steps,over_loss/(steps+1),over_acc/(steps+1),over_pmr/(steps+1))))

        train_steps = steps+1
    return over_loss/train_steps,over_acc/train_steps,over_pmr/train_steps





def mask_out(out,sentence_num_list,train_labels):
    mask=gen_mask(out,sentence_num_list)
    mask_out=torch.where(mask,out,train_labels)
    return mask_out


def save_best_model(epoch,val_acc,val_pmr,best_iter,best_score,model,args,optimizer,criterion):
    if args.metric=="pmr":
        score=val_pmr 
    else:
        score=val_acc
    if score>best_score:
        best_score=score
        best_iter=epoch
         
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
        logger.info(f"save best model at epoch {epoch} with score {score}")
    return best_iter,best_score

def validate(model,val_dataloader,device,criterion,epoch,train_loss,train_acc,train_pmr):
    val_loss = 0
    val_acc = 0
    val_pmr=0
    val_steps = 0
    model.eval()
    for steps, (input_ids, attn_masks,sentence_num_list, val_labels) in enumerate(tqdm(val_dataloader)):

    
        val_labels = val_labels.to(device)

        with torch.no_grad():

            out = model(input_ids, attn_masks,sentence_num_list)      
            masked_out=mask_out(out,sentence_num_list,val_labels.float())      
            loss = criterion(masked_out , val_labels.float()).mean()
            val_loss += loss
            pmr,acc=my_metric_fn(val_labels, out,sentence_num_list)
            val_acc +=acc
            val_pmr+=pmr

        val_steps = steps+1
    c = (('Epoch: %1.1f, Training Loss: % 2.5f, Training Accuracy % 3.4f, Training Pmr % 3.4f%% , Validation Loss: % 4.5f, Validation Accuracy % 5.4f, Validation Pmr % 3.4f%% ' %(epoch,train_loss,train_acc,train_pmr, val_loss/val_steps,val_acc/val_steps,val_pmr/val_steps)))
    print(c)       
    logger.info(c)
    return val_loss/val_steps, val_acc/val_steps, val_pmr/val_steps

            
         