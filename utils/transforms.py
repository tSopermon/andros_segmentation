import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(image_size, use_augmentation=False):
    if use_augmentation:
        return A.Compose([
            A.PadIfNeeded(min_height=image_size, min_width=image_size),
            A.RandomCrop(image_size, image_size),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
            ], is_check_shapes=False)
    else:
        return A.Compose([
            A.PadIfNeeded(min_height=image_size, min_width=image_size),
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], is_check_shapes=False)

def get_val_transform(image_size):
    return A.Compose([
        A.PadIfNeeded(min_height=image_size, min_width=image_size),
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], is_check_shapes=False)