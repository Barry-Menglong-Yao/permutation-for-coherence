import torch 
import torchsort 
   
from utils.mask import gen_mask

def gen_rank_func():

    # from https://github.com/teddykoker/torchsort
    
    return torchsort.soft_rank


    


    # fast-soft-sort from https://github.com/google-research/fast-soft-sort


def rank_for_variable_length(values, regularization="l2", regularization_strength=1.0):
    ranked_order_list=[]
    for value in values:
        value=torch.unsqueeze(value,0)
        ranked_order=torchsort.soft_rank(value , regularization , regularization_strength )
        ranked_order=torch.squeeze(ranked_order)
        ranked_order_list.append(ranked_order)
    ranked_order_tensor=torch.stack(ranked_order_list, 0)
    
    return ranked_order_tensor

# def ranks(inputs, axis=-1):
	  
#     return torch.argsort(torch.argsort(inputs, descending=False, dim=1), dim=1) 



# def my_metric_fn(y_true, y_pred,sentence_num_list):
#     y_pred = y_pred.to("cpu")
#     y_true = y_true.to("cpu")
#     cat_pred = ranks(y_pred)
#     sub = torch.subtract(y_true ,cat_pred )
    
#     sub_zero=torch.zeros_like(sub)
#     mask=gen_mask(sub,sentence_num_list)
#     sub = torch.where(mask,sub,sub_zero)
    
    
#     pmr=gen_pmr(sub )
#     total=torch.sum(sentence_num_list)
#     acc=(total-torch.count_nonzero(sub ))/total

    


#     return  pmr,acc # Note the `axis=-1`


# def gen_pmr(sub ):
#     var_batch = torch.tensor(sub.shape[0], dtype=torch.int64,  requires_grad = False)
#     main_res = torch.zeros([1], dtype=torch.int64 , requires_grad = False)
#     zero = torch.zeros([1], dtype=torch.int64 , requires_grad = False)
#     one = torch.ones([1], dtype=torch.int64,  requires_grad = False)
#     for i in range(sub.shape[0]):
#         res = torch.eq(zero, torch.count_nonzero(sub[i]))
#         if (res.item()):
#             main_res = torch.add(main_res, one)
#     return torch.divide(main_res,var_batch)


