import torch
from torch.utils.data import Dataset

class NetCDFDataset(Dataset):

    def __init__(self, dataset, test_split=0, validation_split=0, is_validation=False, is_test=False, is_2d_model=False, first_output_channels=None):
        super(NetCDFDataset, self).__init__()
        
        self.is_2d_model = is_2d_model
        # orignal data format batch x time x latitude x longitude x channel
        sr = Splitter(test_split, validation_split)
        if (is_test):
            self.should_flip = DataFlipChecker.check_flip(dataset)
            data = sr.split_test(dataset)
        elif (is_validation):
            data = sr.split_validation(dataset)
        else:
            data = sr.split_train(dataset)
        
        # data format batch x channel x time x latitude x longitude
        self.X = torch.from_numpy(data.x.values).float().permute(0, 4, 1, 2, 3)
        self.X = self.X[:, :, :5, :, :]
        
        if first_output_channels is not None:
            y_data = data.y.values[:, :, :, :, :first_output_channels]
            self.y = torch.from_numpy(y_data).float().permute(0, 4, 1, 2, 3)
        else:
            self.y = torch.from_numpy(data.y.values).float().permute(0, 4, 1, 2, 3)
        
        if self.is_2d_model:
            self.X = torch.squeeze(self.X)
            self.y = torch.squeeze(self.y)
                  
        del data
        
    def __getitem__(self, index):
        return (self.X[index,:,:,:,:], self.y[index])

    def __len__(self):
        return self.X.shape[0]
  
   
    
class Splitter():
    def __init__(self, test_rate, validation_rate):
        self.test_rate = test_rate
        self.validation_rate = validation_rate / (1. - self.test_rate)
        
    def split_test(self, dataset):
        return self.__split(dataset, self.test_rate, first_part=False)
        
    def split_validation(self, dataset):
        data = self.__split(dataset, self.test_rate, first_part=True)
        return self.__split(data, self.validation_rate, first_part=False)
        
    def split_train(self, dataset):
        data = self.__split(dataset, self.test_rate, first_part=True)
        return self.__split(data, self.validation_rate, first_part=True)
    
    def __split(self, dataset, split_rate, first_part):
        if (split_rate):
            split = int(dataset.sample.size * (1. - split_rate))
            if (first_part):
                return dataset[dict(sample=slice(0,split))]
            else:
                return dataset[dict(sample=slice(split, None))]
            

class DataFlipChecker():
    """
    Checks if geographic data needs to be flipped for proper visualization.
    
    When using matplotlib's imshow(), the origin is at the top-left by default (origin='upper').
    For geographic data with lat/lon coordinates, this can cause the map to appear upside down.
    """
    
    @staticmethod
    def check_flip(dataset):
        # Check if the object has coordinates attribute (xarray.Dataset)
        if not hasattr(dataset, 'coords'):
            return False
        
        lat_coord, lon_coord = None, None
        
        for coord_name in dataset.coords:
            coord_lower = coord_name.lower()
            
            if 'lat' in coord_lower:
                lat_coord = coord_name
            elif 'lon' in coord_lower:
                lon_coord = coord_name
        
        if lat_coord is not None and lon_coord is not None:
            return True
        
        return False