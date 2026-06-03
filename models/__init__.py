from .unet import UNet
from .attention_unet import AttentionUNet
from .resunet import ResUNet

def get_model(name, **kwargs):
    if name == "unet":
        return UNet(**kwargs)
    elif name == "attention_unet":
        return AttentionUNet(**kwargs)
    elif name == "resunet":
        return ResUNet(**kwargs)
    elif name == "smp_resnet18":
        import segmentation_models_pytorch as smp
        return smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
