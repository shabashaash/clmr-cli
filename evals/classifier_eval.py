import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import classification_report

def stable_encoding(batch, encoder):
    h0 = []
    ld = DataLoader(batch, batch_size=16)
    for cutted in ld:
        for out in encoder(cutted):
            h0.append(out)
    h0 = torch.stack(h0)
    h0 = torch.squeeze(h0, 1)

    return h0


def accuracy_f(output, target, topk=(1,2,5)):
    maxk = max(topk)
    topkeys, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    topkeys = topkeys.numpy().tolist()
    
    top_keys = {}
    for i,v in enumerate(pred.numpy().tolist()):
        top_keys[i] = dict(zip(v, topkeys[i]))
    
    ret = []
    for k in topk:
        correct = (target * torch.zeros_like(target).scatter(1, pred[:, :k], 1)).float()
        ret.append( (correct.sum() / target.sum()).item() )
    return ret, top_keys


def accuracy_f2(output, names, tags_inv, topk=(1,2,5)):
    maxk = max(topk)
    topkeys, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    topkeys = topkeys.numpy().tolist()
    top_keys = {}
    for i,v in enumerate(pred.numpy().tolist()):
        top_keys[names[i]] = [tags_inv[tag_idx] for tag_idx in v]
    return top_keys


def c_evaluate(
    encoder, 
    finetuned_head,
    eval_dataset,
    device,
    r_st_labels
) -> dict:
    est_array = []
    gt_array = []

    encoder = encoder.to(device)
    encoder.eval()

    if finetuned_head is not None:
        finetuned_head = finetuned_head.to(device)
        finetuned_head.eval()




    with torch.no_grad():
        names = []
        for batch, label, name in tqdm(eval_dataset):
            batch = batch.to(device)
            output = stable_encoding(batch, encoder)
            if finetuned_head:
                output = finetuned_head(output)

            output = torch.sigmoid(output)


            track_prediction = output.mean(dim=0)
            est_array.append(track_prediction)
            gt_array.append(label)
            names.append(name)
    
    
    est_array = torch.stack(est_array, dim=0).cpu()
    gt_array = torch.stack(gt_array, dim=0)
    print('before_custom')
    print(est_array, gt_array, "EST_GT")
    top_ks_scores, top_ks = accuracy_f(est_array, gt_array)
    print(top_ks)
    print(gt_array.topk(5, dim=1, largest=True, sorted=True))
    
    
    top_ks = accuracy_f2(est_array, names, r_st_labels)
    print(top_ks,'f2 ACC')
    est_array = (est_array > 0.5).int()

    print(metrics.average_precision_score(gt_array, est_array, average="macro"), "pr_auc")
    
    
    print(classification_report(gt_array, est_array, target_names=list(r_st_labels.values())))
    
    return {'1,2,5 | ':top_ks_scores}