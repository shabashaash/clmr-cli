import os



import torch
from torchaudio_augmentations import Compose, RandomResizedCrop
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer


from .models import SampleCNN


from .utils import (
    load_encoder_checkpoint, 
    load_finetuner_checkpoint
)
from collections import OrderedDict



from simclr import SimCLR

from .modules import RegressionRecomend


from data import ContrastiveDataset
from datasets.recomendations import RECOMENDATIONS
from datasets.audio import AUDIO
from datasets.eval_dataset import EvalDataset
from datasets.test_dataset import TestDataset
from torch.utils.data import DataLoader


from evals.recomend_eval import evaluate
from evals.recomend_predict import predict

#------------------------------------

def main(args):
    pl.seed_everything(args.seed)

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError("That checkpoint (encoder) does not exist")

    if (not os.path.exists(args.playlist_paths) or len(glob(args.playlist_paths+"/*")) == 0) and not os.path.exists(args.labels_file_path):
        raise FileNotFoundError("Playlists folder does not exist or empty.")


    st_labels = {}
    r_st_labels = {}

    if os.path.exists(args.labels_file_path):
        for i,v in enumerate(open(args.labels_file_path, encoding="utf-8").read().split('/n')):
            st_labels[i] = v
            r_st_labels[v] = i
    else:
        for i,v in enumerate(glob(args.playlist_paths+"/*")):
            st_labels[i] = v
            r_st_labels[v] = i


    encoder = SampleCNN(
        strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised=args.supervised
    )

    n_features = encoder.fc.in_features  

    state_dict = load_encoder_checkpoint(args.checkpoint_path)
    encoder.load_state_dict(state_dict)
    encoder = SimCLR(encoder, args.projection_dim, n_features)


    

    if not os.path.exists(args.classifier_checkpoint_path) and not args.train:
        raise FileNotFoundError("That checkpoint (classifier) does not exist, can't predict.")


    state_dict = load_finetuner_checkpoint(args.classifier_checkpoint_path)

    classes_count = list(state_dict.values())[-1].shape[0]

    if classes_count != len(st_labels):
        raise ValueError("Classes count from checkpoint != provided classes count.")



    module = RegressionRecomend(
        args,
        encoder,
        hidden_dim = n_features,
        classes_count = classes_count
    )
    print("Loaded encoder checkpoint.")

    module.classifier_head.load_state_dict(state_dict)
    print("Loaded classifier checkpoint.")

    module.classifier_head.eval()
    module.classifier_head.requires_grad_(False)

    module.encoder.requires_grad_(False)
    module.encoder.eval()

    if args.recomender_checkpoint_path:
        state_dict = OrderedDict({
            k:v
            for k,v in torch.load(args.recomender_checkpoint_path, map_location=torch.device(args.accelerator))["state_dict"].items()
        })
        module.load_state_dict(state_dict)
        print("Loaded recomender checkpoint.")
    
    if args.mode == "train":
        train_transform = [RandomResizedCrop(n_samples=args.audio_length)]

        train_dataset = RECOMENDATIONS(root = args.dataset, subset="full", playlist_path = args.playlist_path, t_mode = args.t_mode, src_ext_audio = args.src_ext_audio)
        valid_dataset = RECOMENDATIONS(root = args.dataset, subset="valid", playlist_path = args.playlist_path, t_mode = args.t_mode, src_ext_audio = args.src_ext_audio)
        
        contrastive_train_dataset = ContrastiveDataset(
            train_dataset,
            input_shape=(1, args.audio_length),
            transform=Compose(train_transform),
        )

        contrastive_valid_dataset = ContrastiveDataset(
            valid_dataset,
            input_shape=(1, args.audio_length),
            transform=Compose(train_transform),
        )


        train_loader = DataLoader(
            contrastive_train_dataset,
            batch_size=32,#args.finetuner_batch_size,
            num_workers=args.workers,
            shuffle=True,
        )

        valid_loader = DataLoader(
            contrastive_valid_dataset,
            batch_size=32,#args.finetuner_batch_size,
            num_workers=args.workers,
            shuffle=True,
        )


        train_representations_dataset = module.extract_representations(train_loader)
        train_loader = DataLoader(
            train_representations_dataset,
            batch_size=32,#args.batch_size,
            num_workers=args.workers,
            shuffle=True,
        )

        valid_representations_dataset = module.extract_representations(valid_loader)
        valid_loader = DataLoader(
            valid_representations_dataset,
            batch_size=32,#args.batch_size,
            num_workers=args.workers,
            shuffle=False,
        )
        early_stop_callback = EarlyStopping(
            monitor="Valid/loss", patience=20, verbose=False, mode="min"
        )
        trainer = Trainer.from_argparse_args(
            args,
            max_epochs=-1, #args.finetuner_max_epochs,
            callbacks=[early_stop_callback],
            accelerator=args.accelerator, 
            log_every_n_steps=1,
        )
        print("Started training recomender.")
        trainer.fit(module, train_loader, valid_loader)
    elif args.mode == "eval":
        print("Evaluating recomend.")
        eval_dataset = RECOMENDATIONS(root = args.dataset, subset="test", playlist_path = args.playlist_path, t_mode = args.t_mode, src_ext_audio = args.src_ext_audio)
        eval_dataset = EvalDataset(
            test_dataset,
            input_shape=(1, args.audio_length),
            transform=None,
        )
        print( r_evaluate(
            module,
            eval_dataset,
            device=args.accelerator
        ) )
    elif args.mode == "predict":
        predict_dataset = AUDIO(root = args.predict_folder, src_ext_audio = args.src_ext_audio)
        predict_dataset = TestDataset(
            predict_dataset,
            input_shape=(1, args.audio_length),
            transform=None,
        )

        print( r_predict(
            module,
            predict_dataset,
            args.accelerator,
            args.model,
            args.topK,
            r_st_labels
        ))


        
            



        