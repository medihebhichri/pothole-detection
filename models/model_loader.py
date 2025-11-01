import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def load_model(checkpoint_path, device='cpu'):
    """Load DeepLabV3+ model"""
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=2
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✅ Model loaded: {checkpoint.get('model_name', 'Unknown')}")
    print(f"   Validation IoU: {checkpoint.get('val_iou', 'N/A')}")

    return model


def load_unet_resnet34(checkpoint_path, device='cpu'):
    """Load U-Net with ResNet34 backbone"""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=2
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def load_fpn_efficientnet(checkpoint_path, device='cpu'):
    """Load FPN with EfficientNet-B0"""
    model = smp.FPN(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=2
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def load_custom_unet(checkpoint_path, device='cpu'):
    """Load Custom U-Net architecture"""
    from models.custom_architectures import CustomUNet

    model = CustomUNet(in_channels=3, num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model