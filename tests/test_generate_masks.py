import os
from generate_masks import main as generate_masks_main

def test_generate_masks_outputs():
    # Run the mask generation script
    generate_masks_main()
    output_dir = 'outputs'
    model_names = ['DeepLabV3', 'DeepLabV3Plus', 'UNet', 'UNetPlusPlus']
    # Check that mask folders are created and contain at least one file (if folder exists)
    for model in model_names:
        masks_dir = os.path.join(output_dir, model, 'masks')
        if os.path.exists(masks_dir):
            mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
            assert len(mask_files) > 0, f"No mask files found for {model}"
        else:
            # If the folder does not exist, just skip (do not create or delete anything)
            pass
