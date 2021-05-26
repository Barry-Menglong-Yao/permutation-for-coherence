from model.model import *
import torch
from utils.fenchel_young_loss import * 
from dba.data_dealer import * 
from model.layers.differential_ranking import * 
from pathlib import Path
from utils.saver import * 

def test(args,  checkpoint):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model = Network(args.d_mlp)
    model=model.to(device)
    checkpoint = torch.load("/content/checkpoints_pytorch/model_0")
    model.load_state_dict(checkpoint['model_state_dict'])
     
def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters()]))
 
def train(args):
    train_dataloader,val_dataloader=load_data(args)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model = Network(args.d_mlp)
    model=model.to(device)
    print_params(model)
    print(model)
    criterion = FenchelYoungLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6) 
    if args.amp=='Y':
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    from tqdm import tqdm
    for epoch in range(args.max_epochs):

        acc = 0
        over_loss =0
        train_steps = 0 
        lr = 5e-7
        if epoch<1:
            lr = 5e-7
        else:
            lr = 1e-7
        for g in optimizer.param_groups:
            g['lr'] = lr
        print("Learning Rate is: \n", lr)
        
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

            
        model.train()
        print("\n EPOCH DONE \n")
        c = (('Epoch: %1.1f, Training Loss: % 2.5f, Training Accuracy % 3.4f, Validation Loss: % 4.5f, Validation Accuracy % 5.4f ' %(epoch, over_loss/(train_steps),acc/(train_steps), val_loss/val_steps, val_acc/val_steps)))
        print(c)        
        print("\nSaving Model")
        main_path = Path(args.output_parent_path)
        model_path, log_path=get_output_path_str(main_path)
         
        PATH = model_path+'/model_'+str(epoch)

        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion
                }, PATH)
    

    


