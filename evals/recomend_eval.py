import torch
from torch.utils.data import DataLoader
import torchmetrics
import torchmetrics.functional as mf
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




def r_evaluate(
    whole_model,
    eval_dataset,
    device
) -> dict:
    est_array = []
    gt_array = []
    names = []
    
    whole_model = whole_model.to(device)
    whole_model.eval()
    whole_model.freeze()
    whole_model.encoder.eval()
    whole_model.classifier_head.eval()
    
    
    
    
    average_precision = torchmetrics.AveragePrecision(pos_label=1) #!

    
    with torch.no_grad():
        for batch, label, name in tqdm(eval_dataset):        
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
            est_array.append(track_prediction)
            gt_array.append(label)
            names.append(name)

    est_array = torch.stack(est_array, dim=0).cpu()
    
    gt_array = torch.stack(gt_array, dim=0).cpu().int()    #.squeeze()#.numpy()


   
    #--------------------------------BCE-------------------------------
#     est_array = (est_array > 0.5).int()
    #--------------------------------BCE-------------------------------

    for name, pred, true in zip(names, est_array, gt_array):
        print(f"Name: {name}, Prediction: {pred}, True: {true}")
    
    
    
    r2score = mf.r2_score(est_array, gt_array)
    msle = mf.mean_squared_log_error(est_array, gt_array)
    ap = average_precision(est_array, gt_array)
    pr = mf.precision(est_array, gt_array, average = "macro", num_classes=2,multiclass=True)
    rc = mf.recall(est_array, gt_array, average = "macro", num_classes=2,multiclass=True)

    
    return {
        "msle": msle,
        "r2score": r2score,
        "average_precision":ap,
        "precision":pr,
        "recall":rc
    }