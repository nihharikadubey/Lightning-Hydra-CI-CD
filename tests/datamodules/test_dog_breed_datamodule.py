import pytest
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.datamodules.dog_breed_datamodule import DogBreedImageDataModule

@pytest.fixture
def datamodule():
    return DogBreedImageDataModule(batch_size=32, num_workers=0)

def test_dog_breed_datamodule_setup(datamodule):
    datamodule.prepare_data()
    datamodule.setup()
    
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None
    
    assert len(datamodule.train_dataset) > 0
    assert len(datamodule.val_dataset) > 0
    assert len(datamodule.test_dataset) > 0
    
    total_samples = sum([len(ds) for ds in [datamodule.train_dataset, datamodule.val_dataset, datamodule.test_dataset]])
    assert total_samples == len(datamodule._dataset)

def test_dog_breed_datamodule_train_dataloader(datamodule):
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    batch = next(iter(train_dataloader))
    assert len(batch) == 2  # (x, y)
    assert batch[0].shape == (32, 3, 224, 224)
    assert batch[1].shape == (32,)

def test_dog_breed_datamodule_val_dataloader(datamodule):
    datamodule.setup()
    val_dataloader = datamodule.val_dataloader()
    batch = next(iter(val_dataloader))
    assert len(batch) == 2  # (x, y)
    assert batch[0].shape == (32, 3, 224, 224)
