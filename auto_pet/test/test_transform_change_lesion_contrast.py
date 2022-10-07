import os
import unittest
from PIL import Image
import matplotlib.pyplot as plt
import torch
from auto_pet.projects.segmentation.preprocessing.preprocess_hdf5 import read_case_hdf5
from auto_pet.projects.segmentation.transform_change_lesion_contrast import TransformChangeLesionContrast
import numpy as np
from config import export_path, data_v13_root


class TestTransformChangeLesionContrast(unittest.TestCase):
    def test_function_to_test(self):
        data = 'PETCT_30c4b7062b_tp_01-18-2001-NA-P.pkl.lz4'

        path = os.path.join(data_v13_root, data)
        case_data = read_case_hdf5(path)
        augmenter = TransformChangeLesionContrast(probability=1.0)

        color_map = plt.get_cmap('binary')
        max_value = 7
        suv_mip = []
        for i in range(20):
            case_data_augmented = augmenter(case_data)
            mip, _ = case_data_augmented['suv'].max(dim=1)
            # make sure the intensities are comparable
            mip = torch.clip(mip, 0, max_value).numpy()
            mip = color_map(mip / max_value)[:, :, :3] * 255.0
            mip = np.flipud(mip)
            suv_mip.append(mip)
        

        suv_mip_images = [Image.fromarray(i.astype(np.uint8)) for i in suv_mip]
        suv_mip_images[0].save(
            fp=os.path.join(export_path, 'lesion_contrast_augmentations.gif'), 
            format='GIF', 
            append_images=suv_mip_images, 
            save_all=True, 
            duration=400, 
            loop=0
        )


if __name__ == '__main__':
    unittest.main()