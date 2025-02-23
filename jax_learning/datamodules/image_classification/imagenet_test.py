import pytest

from jax_learning.conftest import setup_with_overrides
from jax_learning.datamodules.image_classification.image_classification_test import (
    ImageClassificationDataModuleTests,
)
from jax_learning.datamodules.image_classification.imagenet import ImageNetDataModule
from jax_learning.utils.testutils import needs_network_dataset_dir


@pytest.mark.slow
@needs_network_dataset_dir("imagenet")
@setup_with_overrides("datamodule=imagenet")
class TestImageNetDataModule(ImageClassificationDataModuleTests[ImageNetDataModule]): ...
