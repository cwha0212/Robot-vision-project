from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
  def __init__(self, image, mode, transform, num_val=100):
    self.mode = mode
    self.transform = transform
    self.image = image
    self.num_test = 1

    self.test_filenames = []
    self.test_poses = []
    self.train_filenames = []
    self.train_poses = []

  def __getitem__(self, index):
    image = self.image
    
    return self.transform(image)

  def __len__(self):
    num_data = self.num_test
    return num_data



def get_loader(image, model, batch_size, is_shuffle=False, num_val=100):

  if model == 'Googlenet':
    img_size = 300
    img_crop = 299
  elif model == 'Resnet':
    img_size = 256
    img_crop = 224

  transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_crop),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

  batch_size = 1
  is_shuffle = False
  dataset = CustomDataset(image,'test', transform)
  data_loaders = DataLoader(dataset, batch_size, is_shuffle, num_workers=4)
  assert 'Unavailable Mode'

  return data_loaders