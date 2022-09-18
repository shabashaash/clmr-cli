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
    h0 = torch.squeeze(h0, 1)

    return h0

def accuracy_f2(output, names, st_labels, topk=(1,2,5)):
    maxk = max(topk)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    top_keys = {}
    for i,v in enumerate(pred.numpy().tolist()):
        top_keys[names[i]] = [st_labels[tag_idx] for tag_idx in v]
    return top_keys


def r_predict(
    whole_model,
    test_dataset,
    device,
    model,
    topK,
    st_labels
) -> dict:
    est_array = []
    names = []
    
    whole_model = whole_model.to(device)
    whole_model.eval()
    whole_model.freeze()
    whole_model.encoder.eval()
    whole_model.classifier_head.eval()


    
    with torch.no_grad():
        for batch, name in tqdm(test_dataset):        
            batch = batch.to(device)
            h0 = stable_encoding(batch, whole_model.encoder)

#!--------------------------------------------------------------------!            
#             classes = whole_model.classifier_head(h0)
        
#             classes = torch.sigmoid(classes)
            
#             h = torch.cat([h0,classes],dim=1)
#!--------------------------------------------------------------------!        
            
    
    
#!--------------------------------------------------------------------!
            classes = whole_model.classifier_head(h0)
        
            classes = torch.sigmoid(classes)


#             table_coded = whole_model.table_model(classes)
#             audio_coded = whole_model.audio_model(h0)

            h = torch.cat([classes,h0],dim=1) #-1)
            
#!--------------------------------------------------------------------!           
    
            track_prediction = torch.sigmoid(whole_model.model(h)).mean(dim=0) #output.mean(dim=0)

            if model == "full":
                est_array.append([classes, track_prediction])
            elif model == "recomend":
                est_array.append(track_prediction)

            names.append(name)

    est_array = torch.stack(est_array, dim=0).cpu()   


    if model == "full":
        top_ks = accuracy_f2(est_array[:,0], names, st_labels, (topK,) )
        print(top_ks)
    else:
        for name, pred in zip(names, est_array):
            print(f"Name: {name}, Prediction: {pred}")

    return dict(zip(names, est_array))