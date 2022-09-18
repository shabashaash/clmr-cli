from torch import Tensor, FloatTensor
from typing import Tuple
from glob import glob
from .dataset import Dataset
import random
# import fpl_reader



class PLAYLISTS(Dataset):
    def __init__(
        self,
        root,
        playlist_paths: str = "",
        src_ext_audio: str = ".wav",
        st_labels: dict = {},
        subset:str = "train"
    ) -> None:  
        
        super(PLAYLISTS, self).__init__(root)
        self._src_ext_audio = src_ext_audio
        self._playlist_paths = glob(playlist_paths+"/*")
        self.st_labels = st_labels

        self.true_paths = set(glob(root+"/*"+src_ext_audio))

        self.data = {}
        
        self.subset = subset
        
        
        train_valid_split = 0.9
        
        test_store = {}
        
        
        
        # print(len(self.true_paths))
        
        # for i in glob("../../input/trackswav/converted/*.wav"):  #8gb
        #     self.true_paths.add(i)
        
        # print(len(self.true_paths))
        
        
        
        
        
        tracks = [i.split('/')[-1].split('-converted')[0] for i in self.true_paths]

        tmp_paths = []

        for name in tracks:
            words_local = []
            for word in name.split(' '):

                tmp_ = ''.join(e for e in word if e.isalnum())
                
                # if 'umib' not in tmp_:
                tmp_ = ''.join(e for e in tmp_ if not e.isdigit())

                tmp_ = ' '.join(tmp_.split())

                words_local.append(tmp_)

            tmp_paths.append(' '.join([i for i in words_local if len(i)>0]))

        data_dict = dict(zip(tmp_paths, self.true_paths))
        
        
        print(len(data_dict), "FolderPaths")

        for playlist in self._playlist_paths:
            with open(playlist, encoding = "utf-8") as f:
                for line in f.read().split(';'):
                    if len(line)>0:

                        words_local = []
                        for word in line.split(' '):

                            tmp_ = ''.join(e for e in word if e.isalnum())
                            
                            
                            # if 'umib' not in tmp_:
                            tmp_ = ''.join(e for e in tmp_ if not e.isdigit())

                            tmp_ = ' '.join(tmp_.split())

                            words_local.append(tmp_)

                        line_ = ' '.join([i for i in words_local if len(i)>0])
                        if line_ in data_dict.keys():
                            if data_dict[line_] in self.data:
                                self.data[data_dict[line_]].add(playlist.split('/')[-1][:-4])
                            else:
                                self.data[data_dict[line_]] = set()
                                self.data[data_dict[line_]].add(playlist.split('/')[-1][:-4])

        self.data = list(self.data.items())
        self.n_classes = len(self.st_labels)
        
        
        if subset == "train":
            self.data = self.data[:int(train_valid_split*len(self.data))]
        if subset == "valid":
            self.data = self.data[int(train_valid_split*len(self.data))+1:]
        if subset=='test':       
            self.data = random.sample(self.data, 200) #17 66
            print(self.data,'TESTFINALE')

            
            
        print(subset, len(self.data), self.st_labels, len(self.st_labels))
        if len(self.data) == 0:
            raise RuntimeError(
                "Dataset not found. Please place the audio files in the {} folder.".format(
                    root
                )
            )

    def file_path(self, n: int) -> str:
        fp = self.data[n][0]
        return fp

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            Tuple [Tensor, Tensor]: ``(waveform, label)``
        """
        
        audio, _ = self.load(n)
        label_bin = [0 for _ in range(self.n_classes)]
        labels = self.data[n][1]
        for label in labels:
            label_bin[self.st_labels[label]] = 1
        
        return audio, FloatTensor(label_bin), self.data[n][0]

    def __len__(self) -> int:
        return len(self.data)