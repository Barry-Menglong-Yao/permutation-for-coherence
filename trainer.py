from model.model import *
import torch

def test(args,  checkpoint):
    checkpoint = torch.load("/content/checkpoints_pytorch/model_0")
    model.load_state_dict(checkpoint['model_state_dict'])
     

#TODO 
def train(args):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model = Network(args.d_mlp)
    model=model.to(device)
    criterion = FenchelYoungLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6) 
    from tqdm import tqdm
    for epoch in range(epochs):

        acc = 0
        over_loss =0
        train_steps = 0;

        
        for steps, (input_ids, attn_masks, train_labels) in enumerate(tqdm(my_dataloader)):

            input_ids = input_ids.to("cuda")
            attn_masks = attn_masks.to("cuda")
            train_labels = train_labels.to("cuda")

            out = model(input_ids, attn_masks)

            loss = criterion(out, train_labels.float()).mean()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute accuracy

            over_loss += loss
            acc += my_metric_fn(train_labels, out)

            if steps%1000==0:
                print(('Step: %1.1f, Loss: % 2.5f, Accuracy % 3.4f%%' %(steps,over_loss/(steps+1),acc/(steps+1))))

            train_steps = steps+1;

            
            
        # DO EVAL ON VALIDATION SET

        #DataLoader
        val_dataset = TensorDataset(validation_data_input_ids,validation_data_attention_masks, val_new_labels) 
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True) 


        val_loss = 0;
        val_acc = 0;
        val_steps = 0; 
        for steps, (input_ids, attn_masks, val_labels) in enumerate(tqdm(val_dataloader)):

            input_ids = input_ids.to("cuda")
            attn_masks = attn_masks.to("cuda")
            val_labels = val_labels.to("cuda")

            with torch.no_grad():

                out = model(input_ids, attn_masks)            
                loss = criterion(out, val_labels.float()).mean()
                val_loss += loss
                val_acc += my_metric_fn(val_labels, out)


            val_steps = steps+1;

            

        print("\n EPOCH DONE \n")
        print(('Epoch: %1.1f, Training Loss: % 2.5f, Training Accuracy % 3.4f, Validation Loss: % 4.5f, Validation Accuracy % 5.4f ' %(epoch, over_loss/(train_steps),acc/(train_steps), val_loss/val_steps, val_acc/val_steps)))

        print("\nSaving Model")

        PATH = '/content/checkpoints_pytorch/model_'+str(epoch)

        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion
                }, PATH)
    

    


