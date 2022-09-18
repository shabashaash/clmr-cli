from pytorch_lightning import LightningModule
import torch.nn as nn
import torchmetrics
import torch
import torchmetrics.functional as mf
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset
import itertools
from torch import Tensor   #, FloatTensor
from typing import Tuple


class RegressionRecomend(LightningModule):
    def __init__(self, args, encoder: nn.Module, hidden_dim: int, classes_count: int, embed_hidden_dim:int = 128, alpha:float = 1):
        super().__init__()
        self.save_hyperparameters(args)

        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.classifier_head = nn.Sequential(nn.Linear(self.hidden_dim, classes_count))
        
        
        
        
        
        
#         self.audio_linear = nn.Sequential(
#                 nn.Linear(self.hidden_dim, self.hidden_dim),
# #                 nn.ReLU(),
# #                 nn.Dropout(0.1),
# #                 nn.Linear(self.hidden_dim+classes_count, 1),
#             )
        
        
#         self.classes_layer = nn.Sequential(
#                 nn.Linear(classes_count, classes_count),
# #                 nn.ReLU(),
# #                 nn.Dropout(0.1),
# #                 nn.Linear(self.hidden_dim+classes_count, 1),
#             )
        
        
#         print(self.hidden_dim+classes_count)
        
        
        
        
#         self.model = nn.Sequential(
#                 nn.Linear(self.hidden_dim+classes_count, embed_hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(0.5), #0.5 0.1
#                 nn.Linear(embed_hidden_dim, 1), #2
#             )

#---------------------------------------------canon-mlp-------------------------
#         self.model = nn.Sequential(
#                 nn.Linear(self.hidden_dim, embed_hidden_dim, bias=False),
# #                 nn.LayerNorm(self.hidden_dim),
#                 nn.ReLU(),
# #                 nn.Tanh(),
# #                  nn.Dropout(0.1),
#                  #0.5 0.1
#                 nn.Linear(embed_hidden_dim, 1, bias=False), #2
#             ) #canon-mlp
#---------------------------------------------canon-mlp-------------------------
        
#!------------------double loss----------------------        
    
#         self.classification_head = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim//2, bias = False),
#             nn.ReLU(),
# #             nn.Dropout(0.1),
#             nn.Linear(self.hidden_dim//2, classes_count, bias = False)
#         )
        
#         self.like_head = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim, bias = False),
#             nn.ReLU(),
# #             nn.Dropout(0.1),
#             nn.Linear(self.hidden_dim, 1, bias = False)
#         )
         
#         self.alpha = alpha
        
        
        
#!------------------double loss----------------------        
        
        
        #---------------------------------------------NOT LATE FUSION-----------------------------------------------
        
        
#         self.table_model = nn.Sequential(
#             nn.Linear(classes_count, classes_count//2),
#             nn.ReLU(),
#             nn.Linear(classes_count//2, classes_count//2)
#         )
        
#         self.audio_model = nn.Sequential(
#             nn.Linear(self.hidden_dim, embed_hidden_dim),
# #                 nn.LayerNorm(self.hidden_dim),
#             nn.ReLU(),
#             #nn.Dropout(0.1),
#             nn.Linear(embed_hidden_dim, embed_hidden_dim)
#         )
        
        
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim+classes_count, embed_hidden_dim),
#                 nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_hidden_dim, 1)
        )

        #---------------------------------------------NOT LATE FUSION-----------------------------------------------
        
        
        
        #nn.Sequential(nn.Linear(self.hidden_dim+classes_count, 1))
        
        
#         self.model = nn.Sequential(
#                 nn.Linear(self.hidden_dim+classes_count//2, embed_hidden_dim),
#                 nn.ReLU(),
# #                 nn.Dropout(0.5), #0.5 0.1
#                 nn.Linear(embed_hidden_dim, 1)#2
#             )
        
        
        
        
        
        #попробуем линейный слой чтобы не подгонять трансформер
        
        
#         self.map_for_encoder = nn.Sequential(
#             nn.Linear(in_features=self.hidden_dim, out_features=embed_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(in_features=embed_hidden_dim, out_features=embed_hidden_dim)
# #             nn.LayerNorm(embed_hidden_dim)
#         )
        
        
        
#         print(self.hidden_dim+classes_count//2)
        
#         encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim+classes_count, nhead=3) #пока 2 было !2! , 3, !5!, 7, 23
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        
        
        
        
        #мапить каждую модальность в n vector с батч нормализацией
        
#         self.project_t = nn.Sequential()
#         self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
#         self.project_t.add_module('project_t_activation', self.activation)
#         self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        
        
        
#         self.criterions = self.configure_criterions()
        
        
        self.criterion = self.configure_criterion()
        
        
        self.cosine = torchmetrics.CosineSimilarity()
        self.average_precision = torchmetrics.AveragePrecision(pos_label=1)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
#         likebles, classes_pred, classes_true = self._forward_representations(x, y) 
        preds = self._forward_representations(x, y) 
        
#         loss = self.calculate_double_loss(likebles, y, classes_pred, classes_true)
        
        #print(torch.sigmoid(preds), "PREDS", "\n"*2)
        
        loss = self.criterion(preds, y)

        return loss, preds #preds#likebles
    
    def _forward_representations(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Perform a forward pass using either the representations, or the input data (that we still)
        need to extract the represenations from using our encoder.
        """
        if x.shape[-1] == self.hidden_dim:
            h0 = x
        else:
            with torch.no_grad():
                h0 = self.encoder(x)
                
                
#!--------------------------------------------------------------------                 
#         with torch.no_grad():
#             classes = self.classifier_head(h0)
        
#         classes = torch.sigmoid(classes)        
                
#         h = torch.cat([h0,classes],dim=1)        
#!--------------------------------------------------------------------                 
            
            
            
            
            
        #!--------------------------------------------------------------------        
        with torch.no_grad():
            classes = self.classifier_head(h0)
        
        classes = torch.sigmoid(classes)
        
        
#         table_coded = self.table_model(classes)
#         audio_coded = self.audio_model(h0)
        
        h = torch.cat([classes,h0],dim=1) #-1)
        #!--------------------------------------------------------------------




#         embeded = self.map_for_encoder(h0)[:, None, :]

#         h = self.transformer_encoder(h[:, None, :]).squeeze()

        return torch.sigmoid(self.model(h))
#torch.sigmoid(self.model(h0))

#torch.sigmoid(self.like_head(h0)), torch.sigmoid(self.classification_head(h0)), classes 

#tof.softmax(self.model(combined), dim=1) #torch.sigmoid(self.model(combined)) #(self.model(combined) > 0.5).float() * 1 #self.model(combined)

    def training_step(self, batch, _) -> Tensor:
        x, y = batch
        loss, preds = self.forward(x, y)
        
#         y_bool = y > 0.5
        
#         _, preds = torch.max(preds, 1)
        
#         preds = FloatTensor(np.array(preds.cpu())[:, None])
#         y = y.cpu()
        
    
    
    
    
    
    
    
    
#         preds = torch.sigmoid(preds)



        #----------------------------------------BCE----------------------------------------------------------
#         preds = (preds.cpu() > 0.5).int()
#         y = y.cpu().int()    #.squeeze()
        #----------------------------------------BCE----------------------------------------------------------
        
        
        
        
        
        
        
        
        
#         y = (y > 0.5).float()
        
#         print(preds, y)
        self.log("Train/cosine", self.cosine(preds, y))
        self.log("Train/pr_auc", self.average_precision(preds, y))
        self.log("Train/r2_score", mf.r2_score(preds, y))
        self.log("Train/MSLE", mf.mean_squared_log_error(preds, y))
        
        self.log("Train/precision", mf.precision(preds.cpu(), y.cpu().int(), average = "macro", num_classes=2, multiclass=True))
        
#         self.log("Train/ACC", metrics.accuracy_score(y, preds))
        
    
    
    
        self.log("Train/recall", mf.recall(preds.cpu(), y.cpu().int(), average = "macro", num_classes=2,multiclass=True))
        
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, _) -> Tensor:
        x, y = batch
        loss, preds = self.forward(x, y)
        
#         _, preds = torch.max(preds, 1)
#         preds = FloatTensor(np.array(preds.cpu())[:, None])
#         y = y.cpu()









#         preds = torch.sigmoid(preds)




        #----------------------------------------BCE----------------------------------------------------------
#         preds = (preds.cpu() > 0.5).int()
#         y = y.cpu().int()    #.squeeze()
        #----------------------------------------BCE----------------------------------------------------------
        
        
        
        
        
        
#         y = (y > 0.5).float()
        
        
#         print(preds, y)
        
        
#         print(preds, y, "\n\nVAL!!!!!!!!!!!!!!!!!!!!!!\n\n")
        self.log("Valid/cosine", self.cosine(preds, y))
        self.log("Valid/pr_auc", self.average_precision(preds, y))
        self.log("Valid/r2_score", mf.r2_score(preds, y))
        self.log("Valid/MSLE", mf.mean_squared_log_error(preds, y))
        
        self.log("Valid/precision", mf.precision(preds.cpu(), y.cpu().int(), average = "macro", num_classes=2, multiclass=True))
        
#         self.log("Valid/ACC", metrics.accuracy_score(y, preds))
        
    
    
    
        self.log("Valid/recall", mf.recall(preds.cpu(), y.cpu().int(), average = "macro", num_classes=2,multiclass=True))
        
        
        
        
        self.log("Valid/loss", loss)
        return loss

    def configure_criterions(self) -> nn.Module:
        criterion_1 = nn.BCELoss() #torch.nn.MSELoss() #torch.nn.BCELoss() #torch.nn.BCEWithLogitsLoss()#!! pos_weight=FloatTensor([1.66])!!! #nn.CrossEntropyLoss() #nn.MSELoss() #nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()
        criterion_2 = nn.BCELoss()
        return (criterion_1, criterion_2)

    def configure_criterion(self) -> nn.Module:
        criterion = nn.BCELoss() #torch.nn.MSELoss() #torch.nn.BCELoss() #torch.nn.BCEWithLogitsLoss()#!! pos_weight=FloatTensor([1.66])!!! #nn.CrossEntropyLoss() #nn.MSELoss() #nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()
        return criterion


    def calculate_double_loss(self,likebles_pred, likebles_true, classes_pred, classes_true):
#         print(classes_pred.shape, classes_true.shape)
        classes_loss = self.criterions[0](classes_pred, classes_true)
        binary_loss = self.criterions[1](likebles_pred, likebles_true)
        
        
        
#         self.log("classes_pred", classes_pred)
#         self.log("classes_true", classes_true)
        
        
        self.log("binary_loss", binary_loss)
        self.log("classes_loss", classes_loss)
        
        return binary_loss + self.alpha*classes_loss

    def configure_optimizers(self) -> dict:
        
        
        
        #!!!!!!!!!!!!В ТРАНСФОРМЕРЕ И ЭНКОДЕРАХ ОПТИМИЗИРОВАЛСЯ ТОЛЬКО ЛИНЕЙНЫЙ ВЫХОДНОЙ СЛОЙ model!!!!!!! ВСЁ ПЕРЕДЕЛАТЬ!!!!!!
        
        #self.table_model.parameters(), self.audio_model.parameters(), self.model.parameters()
        #self.table_model.parameters(), self.model.parameters(), self.transformer_encoder.parameters()
        
        all_params = itertools.chain(self.model.parameters())
        
        
        optimizer = torch.optim.Adam(
            all_params,#all_params,#self.model.parameters(),
            lr=self.hparams.recomender_learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.00001,#0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=True,
        )
        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "Valid/loss",
            }
        else:
            return {"optimizer": optimizer}
    
    def extract_representations(self, dataloader: DataLoader) -> Dataset:

        representations = []
        ys = []
        for x, y in tqdm(dataloader):
            with torch.no_grad():
                h0 = self.encoder(x)
                representations.append(h0)
                ys.append(y)

        if len(representations) > 1:
            representations = torch.cat(representations, dim=0)
            ys = torch.cat(ys, dim=0)
        else:
            representations = representations[0]
            ys = ys[0]

        tensor_dataset = TensorDataset(representations, ys)
        return tensor_dataset