from jax_learning.conftest import setup_with_overrides
from jax_learning.datamodules.image_classification.fashion_mnist import FashionMNISTDataModule
from jax_learning.datamodules.image_classification.image_classification_test import (
    ImageClassificationDataModuleTests,
)


@setup_with_overrides("datamodule=fashion_mnist")
class TestFashionMNISTDataModule(ImageClassificationDataModuleTests[FashionMNISTDataModule]): ...
