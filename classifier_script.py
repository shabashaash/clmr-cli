import os



import torch
from torchaudio_augmentations import Compose, RandomResizedCrop
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint



from models import SampleCNN


from utils import (
    load_encoder_checkpoint, 
    load_finetuner_checkpoint
)
from collections import OrderedDict

from simclr import SimCLR

from modules import LinearEvaluation


from data import ContrastiveDataset
from datasets.playlists import PLAYLISTS
from datasets.audio import AUDIO
from datasets.eval_dataset import EvalDataset
from datasets.test_dataset import TestDataset
from torch.utils.data import DataLoader


from evals.classifier_eval import c_evaluate
from evals.classifier_predict import c_predict

from glob import glob
#------------------------------------

def main(args):
   

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError("That checkpoint (encoder) does not exist")

    if (not os.path.exists(args.playlist_paths) or len(glob(args.playlist_paths+"/*")) == 0) and not os.path.exists(args.labels_file_path):
        raise FileNotFoundError("Playlists folder does not exist or empty.")

    st_labels = {}
    r_st_labels = {}

    if os.path.exists(args.labels_file_path):
        for i,v in enumerate(open(args.labels_file_path, encoding="utf-8").readlines()):
            st_labels[i] = v.rstrip()
            r_st_labels[v.rstrip()] = i
    else:
        for i,v in enumerate(sorted(glob(args.playlist_paths+"/*"))):
            label = v.split('/')[-1].split('.')[0]
            st_labels[i] = label
            r_st_labels[label] = i

    n_classes = len(st_labels)


    print("n_classes", n_classes)

    pl.seed_everything(args.seed)
    encoder = SampleCNN(
        strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised=args.supervised
    )

    n_features = encoder.fc.in_features  

    state_dict = load_encoder_checkpoint(args.checkpoint_path)
    encoder.load_state_dict(state_dict)
    _ = SimCLR(encoder, args.projection_dim, (n_features))
    


    module = LinearEvaluation(
        args,
        encoder,
        hidden_dim=n_features,
        output_dim=n_classes,
    )
    print("Loaded encoder checkpoint.")
    
    module.encoder.requires_grad_(False)
    module.encoder.eval()

    
    if args.classifier_checkpoint_path:
        state_dict = load_finetuner_checkpoint(args.classifier_checkpoint_path)

        classes_count = list(state_dict.values())[-1].shape[0]

        if classes_count != len(st_labels):
            raise ValueError("Classes count from checkpoint != provided classes count.")

        module.model.load_state_dict(state_dict)

    print("Loaded classifier checkpoint.")


    if args.mode == "train":

        
        if not os.path.exists(args.save_checkpoint_path):
            print(f"Can't find save_checkpoint folder {args.save_checkpoint_path}, creating one.")
            os.mkdir(args.save_checkpoint_path)



        train_transform = [RandomResizedCrop(n_samples=args.audio_length)]

        train_dataset = PLAYLISTS(root = args.dataset_dir, subset="full", playlist_paths = args.playlist_paths, st_labels = r_st_labels, src_ext_audio = args.src_ext_audio)
        valid_dataset = PLAYLISTS(root = args.dataset_dir, subset="valid", playlist_paths = args.playlist_paths, st_labels = r_st_labels, src_ext_audio = args.src_ext_audio)
        
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
            batch_size=args.extract_batch_size,
            num_workers=args.workers,
            shuffle=True,
        )

        valid_loader = DataLoader(
            contrastive_valid_dataset,
            batch_size=args.extract_batch_size,
            num_workers=args.workers,
            shuffle=True,
        )


        train_representations_dataset = module.extract_representations(train_loader)
        train_loader = DataLoader(
            train_representations_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=True,
        )

        valid_representations_dataset = module.extract_representations(valid_loader)
        valid_loader = DataLoader(
            valid_representations_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False,
        )
        early_stop_callback = EarlyStopping(
            monitor="Valid/loss", patience=20, verbose=True, mode="min"
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.save_checkpoint_path, save_top_k=1, monitor="Valid/loss"
        )
        trainer = Trainer.from_argparse_args(
            args,
            max_epochs=-1,#args.finetuner_max_epochs,
            callbacks=[early_stop_callback, checkpoint_callback],
            accelerator=args.accelerator, 
            log_every_n_steps=1,
        )
        print("Started training recomender.")
        trainer.fit(module, train_loader, valid_loader)
        #!!python main.py --model "classes" --mode "train" --dataset_dir "/kaggle/input/trackswav/converted" --playlist_paths "/kaggle/input/textplaylists/Converted" --accelerator "cuda:0" --classifier_checkpoint_path "/kaggle/input/mine-checkpoints/finetuner_with18gb_78_711.ckpt" --checkpoint_path "/kaggle/input/mine-checkpoints/encoder_1536_6148.ckpt" --save_checkpoint_path "/kaggle/working/chkpts" (--labels_file_path "/kaggle/working/clmr-cli/labels.txt") 
    
    elif args.mode == "eval":
        print("Evaluating recomend.")
        eval_dataset = PLAYLISTS(root = args.dataset_dir, subset="test", playlist_paths = args.playlist_paths, st_labels = r_st_labels, src_ext_audio = args.src_ext_audio)
        eval_dataset = EvalDataset(
            eval_dataset,
            input_shape=(1, args.audio_length),
            transform=None,
        )
        print( c_evaluate(
            module.encoder,
            module.model,
            eval_dataset,
            args.accelerator,
            st_labels
        ) )
        
    elif args.mode == "predict":
        predict_dataset = AUDIO(root = args.predict_folder, src_ext_audio = args.src_ext_audio)
        predict_dataset = TestDataset(
            predict_dataset,
            input_shape=(1, args.audio_length),
            transform=None,
        )

        print( c_predict(
            module.encoder,
            module.model,
            predict_dataset,
            args.accelerator,
            args.topK,
            st_labels
        ))
