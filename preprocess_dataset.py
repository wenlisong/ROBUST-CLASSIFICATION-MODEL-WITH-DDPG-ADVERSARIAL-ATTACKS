import os
import glob
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from progressbar import *

class ImageSet_preprocess(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        image = Image.open(image_path).convert('RGB')
        image_path = image_path.replace('./datasets/IJCAI_2019_AAAC_train/', './datasets/IJCAI_2019_AAAC_train_processed/')
        _dir, _filename = os.path.split(image_path)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        image.save(image_path)
        return image_path
    
def load_data_jpeg_compression(batch_size=16):
    all_imgs = glob.glob('./datasets/IJCAI_2019/IJCAI_2019_AAAC_train/*/*.jpg')
    train_data = pd.DataFrame({'image_path':all_imgs})
    datasets = {
        'train_data': ImageSet_preprocess(train_data),
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=False) for ds in datasets.keys()
    }
    return dataloaders

if __name__ == '__main__':
    dataloader = load_data_jpeg_compression()
    widgets = ['jpeg :',Percentage(), ' ', Bar('#'),' ', Timer(),
       ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)
    for batch_data in pbar(dataloader['train_data']):
        pass
    # pass