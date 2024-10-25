import pytest
from pathlib import Path
import sys
from omegaconf import OmegaConf
import hydra

# Add the project root to the Python path
#sys.path.append(str(Path(__file__).parent.parent.parent))
# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from  src.data_modules.dogs_datamodule import DogsBreedDataModule
from src.train import train

@pytest.fixture
def config():
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(config_name="train")
    return cfg

def test_dogs_breed_training(config, tmp_path):
    try:
        # Override settings for testing
        config.trainer.max_epochs = 1
        # Remove the following line
        # config.trainer.gpus = 0
        config.data.batch_size = 4
        config.data.num_workers = 0
        config.data.data_dir = str(tmp_path / "data")
        config.paths.output_dir = str(tmp_path / "output")
        config.paths.log_dir = str(tmp_path / "logs")

        # Create dummy data
        (tmp_path / "data" / "train").mkdir(parents=True)
        (tmp_path / "data" / "val").mkdir(parents=True)
        (tmp_path / "data" / "test").mkdir(parents=True)

        # Instantiate trainer, model, and datamodule
        trainer = hydra.utils.instantiate(config.trainer)
        model = hydra.utils.instantiate(config.model)
        datamodule = hydra.utils.instantiate(config.data)

        # Test with different model configurations
        for model_name in ["resnet18", "mobilenetv2"]:
            config.model.name = model_name
            model = hydra.utils.instantiate(config.model)
            train(trainer, model, datamodule)

        # Verify output directory exists and contains files
        output_dir = Path(config.paths.output_dir)
        assert output_dir.exists()
        assert len(list(output_dir.glob('*'))) > 0

        # Verify log directory exists
        assert Path(config.paths.log_dir).exists()



        # =========================================================
        # Check if checkpoints directory exists
        checkpoints_dir = tmp_path / "checkpoints"
        config.trainer.default_root_dir = str(checkpoints_dir)

        # Ensure the checkpoints directory exists
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate components
        datamodule = hydra.utils.instantiate(config.data)
        model = hydra.utils.instantiate(config.model)
        trainer = hydra.utils.instantiate(config.trainer)

        # Run training
        train(config, trainer, model, datamodule)
        
        # Print directory contents for debugging
        print(f"Contents of {tmp_path}:")
        for item in os.listdir(tmp_path):
            print(f"- {item}")

        
        assert checkpoints_dir.exists(), f"Checkpoints directory should be created at {checkpoints_dir}"
        
        # If checkpoints directory exists, check its contents
        if checkpoints_dir.exists():
            print(f"Contents of {checkpoints_dir}:")
            for item in os.listdir(checkpoints_dir):
                print(f"- {item}")
        
        # Add some assertions to verify the training occurred
        assert any(checkpoints_dir.iterdir()), f"At least one checkpoint should be saved in {checkpoints_dir}"

        # Clean up temporary directory after test
        shutil.rmtree(tmp_path) 
        # ===========================================================

    except Exception as e:
        pytest.fail(f"An unexpected error occurred: {str(e)}\nConfig: {OmegaConf.to_yaml(config)}")

def test_train_with_invalid_config():
    with pytest.raises(Exception):
        invalid_config = OmegaConf.create({})
        train(None, None, None)  # This should raise an exception

if __name__ == "__main__":
    pytest.main([__file__])