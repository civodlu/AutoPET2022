from unittest import TestCase
import torch
from trw.layers.gan import GanDataPool


class TestLayersGan(TestCase):
    def test_data_pool(self):
        """
        Test the replacement ratio and insertion ratio of the pool
        """
        pool_size = 1000
        batch_size = 100
        nb_batches_full_pool = pool_size // batch_size

        replacement_probability = 0.35
        insertion_probability = 0.1
        pool = GanDataPool(
            pool_size,
            replacement_probability=replacement_probability,
            insertion_probability=insertion_probability)

        # first, make sure we fully populate the pool before any replacement happens
        for i in range(nb_batches_full_pool):
            batch = {
                'id': torch.arange(i * nb_batches_full_pool, i * nb_batches_full_pool + batch_size)
            }

            images = torch.zeros([batch_size, 1, 16, 16])
            batch_new, images_new = pool.get_data(batch, images)
            assert (batch['id'] == batch_new['id']).all()

        # at this point the pool is full. Test the replacement size
        nb_samples = 0
        nb_replacements = 0
        nb_insertions = 0
        for i in range(100):
            id = torch.arange(nb_samples, nb_samples + batch_size)
            batch = {
                'id': id
            }
            images = torch.ones([batch_size, 1, 16, 16], dtype=torch.float32) * id.type(torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)

            pool_ids_old = torch.tensor([b['id'][0] for b in pool.batches])
            batch_new, images_new = pool.get_data(batch, images)
            pool_ids_new = torch.tensor([b['id'][0] for b in pool.batches])
            nb_replacements += (batch['id'] != batch_new['id']).sum()
            nb_insertions += (pool_ids_old != pool_ids_new).sum()
            nb_samples += batch_size

        effective_replacement_ratio = float(nb_replacements) / nb_samples
        print('effective_replacement_ratio', effective_replacement_ratio)
        assert abs(effective_replacement_ratio - replacement_probability) < 0.02

        effective_insertion_ratio = float(nb_insertions) / nb_samples
        assert abs(effective_insertion_ratio - effective_insertion_ratio) < 0.02
        print('effective_insertion_ratio', effective_insertion_ratio)

