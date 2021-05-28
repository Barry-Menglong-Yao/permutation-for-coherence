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



def test(args,  checkpoint):
    device=DEVICE
    model = Network(args.d_mlp)
    model=model.to(device)
    checkpoint = torch.load("/content/checkpoints_pytorch/model_0")
    model.load_state_dict(checkpoint['model_state_dict'])
     
def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters()]))
 

def gen_model_and_optimizer(args,device):
    model = Network(args.d_mlp)
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
    device=DEVICE
    print(f"use {device}")
    model,optimizer=gen_model_and_optimizer(args,device)
    criterion = FenchelYoungLoss()
    best_score = -np.inf
    best_iter = 0
    for epoch in range(args.max_epochs):
        acc = 0
        over_loss =0
        train_steps = 0 
        model.train()
        for steps, (input_ids, attn_masks, train_labels) in enumerate(tqdm(train_dataloader)):
            input_ids = input_ids.to(device)
            attn_masks = attn_masks.to(device)
            train_labels = train_labels.to(device)

            out = model(input_ids, attn_masks)

            loss = criterion(out.to(device), train_labels.float()).mean()

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
            acc += my_metric_fn(train_labels, out)

            if steps%1000==0:
                print(('Step: %1.1f, Loss: % 2.5f, Accuracy % 3.4f%%' %(steps,over_loss/(steps+1),acc/(steps+1))))

            train_steps = steps+1

            
            
        # DO EVAL ON VALIDATION SET
        val_loss,val_score=validate(model,val_dataloader,device,criterion)
        c = (('Epoch: %1.1f, Training Loss: % 2.5f, Training Accuracy % 3.4f, Validation Loss: % 4.5f, Validation Accuracy % 5.4f ' %(epoch, over_loss/(train_steps),acc/(train_steps), val_loss,val_score)))
        print(c)       
        logger.info(c)

        best_iter,best_score=save_best_model(epoch,val_score,best_iter,best_score,model,args)

        if args.early_stop and (epoch - best_iter) >= args.early_stop:
            print('early stop at epc {}'.format(epoch))
            break
        
    

    
def save_best_model(epoch,score,best_iter,best_score,model,args,optimizer,criterion):
    if score>best_score:
        best_score=score
        best_iter=epoch
        c=f'save best model at epc {epoch}' 
        logger.info(c)
        main_path = Path(args.output_parent_dir)
        model_path, log_path=get_output_path_str(main_path)
        PATH = model_path+'/model_'+args.time_flag

        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "args":args,
                'loss': criterion
                }, PATH)
    return best_iter,best_score

def validate(model,val_dataloader,device,criterion):
    val_loss = 0
    val_acc = 0
    val_steps = 0
    model.eval()
    for steps, (input_ids, attn_masks, val_labels) in enumerate(tqdm(val_dataloader)):

        input_ids = input_ids.to(device)
        attn_masks = attn_masks.to(device)
        val_labels = val_labels.to(device)

        with torch.no_grad():

            out = model(input_ids, attn_masks)            
            loss = criterion(out.to(device), val_labels.float()).mean()
            val_loss += loss
            val_acc += my_metric_fn(val_labels, out)


        val_steps = steps+1
    return val_loss/val_steps, val_acc/val_steps