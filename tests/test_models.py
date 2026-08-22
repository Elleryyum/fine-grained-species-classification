import pytest

from species_classifier.models import build_model, parameter_count


@pytest.mark.parametrize("architecture", ["resnet18", "resnet50"])
def test_model_factory_replaces_classifier(architecture):
    model = build_model(architecture, num_classes=7, pretrained=False)

    assert model.fc.out_features == 7
    assert parameter_count(model) > 1_000_000


def test_model_factory_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="unsupported"):
        build_model("unknown", num_classes=7, pretrained=False)

