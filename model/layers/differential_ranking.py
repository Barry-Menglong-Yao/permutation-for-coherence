import torch 

   


def gen_rank_func():

    # from https://github.com/teddykoker/torchsort
    import torchsort 
    return torchsort.soft_rank

    # fast-soft-sort from https://github.com/google-research/fast-soft-sort



def ranks(inputs, axis=-1):
	  
    return torch.argsort(torch.argsort(inputs, descending=False, dim=1), dim=1)+1



def my_metric_fn(y_true, y_pred):
    y_pred = y_pred.to("cpu")
    y_true = y_true.to("cpu")

    cat_pred = ranks(y_pred)

    zero = torch.zeros([1], dtype=torch.int64 , requires_grad = False)
    one = torch.ones([1], dtype=torch.int64,  requires_grad = False)
    var_batch = torch.tensor(y_pred.shape[0], dtype=torch.int64,  requires_grad = False)
    main_res = torch.zeros([1], dtype=torch.int64 , requires_grad = False)
    for i in range(y_pred.shape[0]):
        sub = torch.subtract(y_true[i,:],cat_pred[i,:])
        res = torch.eq(zero, torch.count_nonzero(sub))
        if (res.item()):
            main_res = torch.add(main_res, one)


    return torch.divide(main_res,var_batch)  # Note the `axis=-1`