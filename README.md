# clmr-cli

# Link to the repo of original model: [https://github.com/Spijkervet/CLMR](https://github.com/Spijkervet/CLMR)

This cli interface supports 3 modes of work (model):

classes - train/eval/predict only tags (multilabel). Tags loaded from playlists folder.

recomend - train/eval/predict only like probabilities (regression). Positive/negative ratings of songs is loaded from playlist file.

full - only for predict. You need to provide encoder, classifier and recomender regressor checkpoint. In this mode model predicts topK tags and probabilities.


## Installing
```
git clone https://github.com/shabashaash/clmr-cli.git && cd clmr-cli

pip3 install -r requirements.txt
```


## Classifier
### Train classifier
```
python main.py --model "classes" --mode "train" --dataset_dir "/dir/to/audio" --playlist_paths "/dir/to/labeled/playlists" --accelerator "cuda:0" or "cpu"  --checkpoint_path "/path/to/encoder/checkpoint" --save_checkpoint_path "/path/where/save/new/classifier" 
Additional: ( 
--labels_file_path "/path/to/tags/file" 
--classifier_checkpoint_path "/path/to/classifier/checkpoint" 
) 
```

### Eval classifier
```
python main.py --model "classes" --mode "eval" --dataset_dir "/dir/to/audio" --playlist_paths "/dir/to/labeled/playlists" --accelerator "cuda:0" or "cpu" --classifier_checkpoint_path "/path/to/classifier/checkpoint" --checkpoint_path "/path/to/encoder/checkpoint"
Additional: ( 
--labels_file_path "/path/to/tags/file"
) 
```
### Predict classifier
```
python main.py --model "classes" --mode "predict" --predict_folder "/path/to/test/audio" --accelerator "cuda:0" or "cpu" --classifier_checkpoint_path "/path/to/classifier/checkpoint" --checkpoint_path "/path/to/encoder/checkpoint"
Additional: ( 
--labels_file_path "/path/to/tags/file"
) 
```

## Recomendations
### Train recomendations
```
python main.py --model "recomend" --mode "train" --dataset_dir "/dir/to/audio" --playlist_path "/dir/to/labeled/playlist" --accelerator "cuda:0" or "cpu" --classifier_checkpoint_path "/path/to/classifier/checkpoint" --checkpoint_path "/path/to/encoder/checkpoint" --save_checkpoint_path "/path/where/save/new/classifier"

Additional: ( 
--labels_file_path "/path/to/tags/file" 
--recomender_checkpoint_path "/path/to/recomender/checkpoint" 
) 
```

### Eval recomendations
```
python main.py --model "recomend" --mode "eval" --dataset_dir "/dir/to/audio" --playlist_path "/dir/to/labeled/playlist" --accelerator "cuda:0" or "cpu" --classifier_checkpoint_path "/path/to/classifier/checkpoint" --checkpoint_path "/path/to/encoder/checkpoint" --recomender_checkpoint_path "/path/to/recomender/checkpoint" 
Additional: ( 
--labels_file_path "/path/to/tags/file"
) 
```
### Predict recomendations
```
python main.py --model "recomend" --mode "predict" --predict_folder "/path/to/test/audio" --accelerator "cuda:0" or "cpu" --classifier_checkpoint_path "/path/to/classifier/checkpoint" --checkpoint_path "/path/to/encoder/checkpoint" --recomender_checkpoint_path "/path/to/recomender/checkpoint"
Additional: ( 
--labels_file_path "/path/to/tags/file"
) 
```

## Predict full (default)
```
python main.py --model "full" --mode "predict" --predict_folder "/path/to/test/audio" --accelerator "cuda:0" or "cpu" --classifier_checkpoint_path "/path/to/classifier/checkpoint" --checkpoint_path "/path/to/encoder/checkpoint" --recomender_checkpoint_path "/path/to/recomender/checkpoint"
Needed to provide one of: ( 
--labels_file_path "/path/to/tags/file"
or 
--playlist_paths "/dir/to/labeled/playlists"
) 
```

## For more in-depth tuning please look at [./config/config.yaml](./config/config.yaml)
