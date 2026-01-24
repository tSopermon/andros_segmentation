import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(image_size, use_augmentation=False):
    if use_augmentation:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.7),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
            ], p=0.8),
            A.OneOf([
                A.GaussNoise(p=1),
                A.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=1),
                A.Blur(blur_limit=3, p=1),
            ], p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1),
                A.ElasticTransform(alpha=1, sigma=50, p=1),
            ], p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
            ], is_check_shapes=False)
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], is_check_shapes=False)

def get_val_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], is_check_shapes=False)