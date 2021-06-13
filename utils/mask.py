import torch
def gen_sentence_mask(  out,sentence_num_list):
    sentence_mask=torch.ones_like(out ,dtype=torch.bool)
    for i in range(len(sentence_num_list)):
        sentence_mask[i,sentence_num_list[i]:]=False
    return sentence_mask