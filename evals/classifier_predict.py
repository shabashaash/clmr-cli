import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def stable_encoding(batch, encoder):
    h0 = []
    ld = DataLoader(batch, batch_size=16)
    for cutted in ld:
        for out in encoder(cutted):
            h0.append(out)
        
    h0 = torch.stack(h0)

    print("h01", h0)

    h0 = torch.squeeze(h0, 1)

    print("h02", h0)

    return h0

def accuracy_f2(output, names, r_st_labels, topk=(1,2,5)):
    maxk = max(topk)
    topkeys, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    topkeys = topkeys.numpy().tolist()
    top_keys = {}
    for i,v in enumerate(pred.numpy().tolist()):
        top_keys[names[i]] = [r_st_labels[tag_idx] for tag_idx in v]
    return top_keys


def c_predict(
    encoder,
    finetuned_head,
    test_dataset,
    device,
    topK,
    r_st_labels
) -> dict:
    est_array = []
    names = []

    encoder = encoder.to(device)
    encoder.eval()
    finetuned_head = finetuned_head.to(device)
    finetuned_head.eval()

    with torch.no_grad():
        for batch, name in tqdm(test_dataset):

            batch = batch.to(device)
            print("batch", batch)
            output = stable_encoding(batch, encoder)
            print("output_ml", output)
            output = finetuned_head(output)
            output = torch.sigmoid(output)
            print("output", output)
            est_array.append(output)
            names.append(name)
    
    
    est_array = torch.stack(est_array, dim=0).cpu()

    print("est_array",est_array)

    top_ks = accuracy_f2(est_array, names, r_st_labels, (topK,))

    for name, pred in zip(names, top_ks):
        print(f"Name: {name}, Prediction: {top_ks}")

    return dict(zip(names, top_ks))