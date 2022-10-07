from functools import partial
import os
import unittest
import tempfile
import numpy as np
from auto_pet.projects.segmentation.preprocessing.preprocess_hdf5 import case_image_sampler_random, read_case_hdf5, write_case_hdf5
from config import export_path

class TestHDF5(unittest.TestCase):
    def test_chunking_coverage(self):
        """
        This is not a real unit test: results
        must be manually looked at. In particular,
        the bottom/right edges should have more samples
        with margin
        """
        with tempfile.TemporaryDirectory() as p:
            batch = {
                'v': np.random.randn(100, 90, 80),
                'current_origin': (0, 0, 0),
                'current_spacing': (1, 1, 1),
            }

            h5_path = p + '/data.hdf5'
            write_case_hdf5(h5_path, batch, (32, 32, 32))

            counts = np.zeros_like(batch['v'])
            for i in range(10000):
                block_shape=(32, 24, 16)
                data = read_case_hdf5(
                    h5_path, 
                    image_names=('v',), 
                    case_image_sampler_fn=partial(case_image_sampler_random, block_shape=block_shape, margin=np.asarray(block_shape) // 2))
                chunk_offset_index = data['chunking_offset_index_zyx']

                if i % 100 == 0:
                    print(f'r={i}, offset={chunk_offset_index}')

                counts[
                    chunk_offset_index[0]:chunk_offset_index[0]+block_shape[0],
                    chunk_offset_index[1]:chunk_offset_index[1]+block_shape[1],
                    chunk_offset_index[2]:chunk_offset_index[2]+block_shape[2]
                ] += 1
            np.save(os.path.join(export_path, 'counts.npy'), counts)
            np.save(os.path.join(export_path, 'data.npy'), batch['v'])

if __name__ == '__main__':
    unittest.main()