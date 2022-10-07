from unittest import TestCase
import torch
import trw
import os
import utils
import pickle
import numpy as np


def functor_extract_i(inputs, batch):
    return batch.get('i')


class TestSimplifiedNN(TestCase):
    @staticmethod
    def create_mnist_net():
        i = trw.simple_layers.Input([None, 1, 28, 28], 'images')
        n = trw.simple_layers.Conv2d(i, out_channels=16, kernel_size=5)
        n = trw.simple_layers.ReLU(n)
        n = trw.simple_layers.MaxPool2d(n, 2, 2)
        n = trw.simple_layers.Flatten(n)
        n = trw.simple_layers.Linear(n, 500)
        n = trw.simple_layers.ReLU(n)
        n = trw.simple_layers.Linear(n, 10)
        n = trw.simple_layers.OutputClassification(n, output_name='softmax', classes_name='targets')
        return n

    def test_parents___mnsit_net(self):
        n = TestSimplifiedNN.create_mnist_net()
        nb_parents = 0
        while n is not None:
            if len(n.parents) == 0:
                n = None
            else:
                assert len(n.parents) == 1
                parent = list(n.parents)[0]
                assert len(parent.children) == 1
                assert list(parent.children)[0]() == n
                n = parent
                nb_parents += 1
        assert nb_parents == 8

    def test_node_output_dependencies__mnsit_net(self):
        n = TestSimplifiedNN.create_mnist_net()
        dependencies = trw.simple_layers.nodes_mark_output_dependencies([n])

        assert len(dependencies) == 9
        for node, ids in dependencies.items():
            assert len(ids) == 1
            assert list(ids)[0] == 0

    def test_find_intput_nodes__mnsit_net(self):
        n = TestSimplifiedNN.create_mnist_net()
        inputs = trw.simple_layers.find_layer_type([n], trw.simple_layers.Input)
        assert len(inputs) == 1

    def test_child_parent(self):
        """
        Make sure the we don't have copies in parents or children
        """
        i = trw.simple_layers.Input([None, 1, 32, 32], 'test')
        l = trw.simple_layers.Conv2d(i, 16, 5)

        assert len(i.children) == 1
        i_child = next(iter(i.children))
        assert i_child() is l

        assert len(l.parents) == 1
        l_parent = next(iter(l.parents))
        assert l_parent is i

    def test_compile__mnsit_net(self):
        n = TestSimplifiedNN.create_mnist_net()
        compiled_nn = trw.simple_layers.compile_nn([n])
        assert len(compiled_nn.inputs) == 1
        assert len(compiled_nn.outputs) == 1
        assert compiled_nn.outputs[0] is n

        # test that we can run and evaluate the compiled network
        inputs = {
            'images': torch.zeros([1, 1, 28, 28]),
            'targets': torch.zeros([1], dtype=torch.long)
        }
        outputs = compiled_nn(inputs)
        results = outputs['softmax'].evaluate_batch(inputs, is_training=False)

        # make sure we have a loss
        assert results['losses'].shape == torch.Size([1])

    def test_compiled_net_pickling(self):
        """
        Make sure we can pickle and unpickle correctly the models
        """
        n = TestSimplifiedNN.create_mnist_net()

        compiled_nn = trw.simple_layers.compile_nn([n])

        temporary = os.path.join(utils.root_output, 'net.pkl')
        with open(temporary, 'wb') as f:
            pickle.dump(compiled_nn, f)

        with open(temporary, 'rb') as f:
            compiled_nn_loaded = pickle.load(f)

        inputs = {
            'images': torch.zeros([1, 1, 28, 28]),
            'targets': torch.zeros([1], dtype=torch.long)
        }
        compiled_nn_output = compiled_nn(inputs)

        compiled_nn_loaded_output = compiled_nn_loaded(inputs)
        output_loaded = compiled_nn_loaded_output['softmax'].output
        output = compiled_nn_output['softmax'].output
        assert (output_loaded == output).all()

    def test_functor_network(self):
        """
        Test using an already existing sequential module: this is
        would facilitate migration to this simple declaration model
        """
        m = trw.layers.convs_2d(input_channels=1, channels=[8, 16], with_flatten=False)

        n = trw.simple_layers.Input([None, 1, 28, 28], 'images')
        n = trw.simple_layers.SimpleModule(n, m)  # the output size will be automatically detected
        n = trw.simple_layers.Flatten(n)
        logit = trw.simple_layers.Linear(n, 10)
        n = trw.simple_layers.OutputClassification(logit, output_name='softmax', classes_name='targets')

        network = trw.simple_layers.compile_nn([n])
        inputs = {
            'images': torch.zeros([1, 1, 28, 28]),
            'targets': torch.zeros([1], dtype=torch.long)
        }
        outputs = network(inputs)
        assert len(outputs) == 1
        
    def test_output_embedding_functor(self):
        n = trw.simple_layers.Input([None, 1], 'i')
        n = trw.simple_layers.OutputEmbedding(n, output_name='record', functor=functor_extract_i)
        network = trw.simple_layers.compile_nn([n])
        
        inputs = {
            'i': torch.from_numpy(np.asarray([0, 1, 2, 3], dtype=np.int)).view([4, 1])
        }
        outputs = network(inputs)
        assert len(outputs) == 1
        assert outputs['record'].output is inputs['i']
        
    def test_simple_module_non_nn_module(self):
        i1 = trw.simple_layers.Input([None, 1], 'i1')
        i2 = trw.simple_layers.Input([None, 1], 'i2')
        n = trw.simple_layers.ConcatChannels([i1, i2])
        n = trw.simple_layers.OutputEmbedding(n, output_name='concat')
        
        network = trw.simple_layers.compile_nn([n])
        inputs = {
            'i1': torch.from_numpy(np.asarray([[1]], dtype=np.float)),
            'i2': torch.from_numpy(np.asarray([[2]], dtype=np.float))
        }
        outputs = network(inputs)
        
        # the order is important!
        assert len(outputs) == 1
        print(outputs['concat'].output)
        assert (outputs['concat'].output == torch.from_numpy(np.asarray([[1, 2]], dtype=np.float))).all()

    def test_simple_module_non_nn_module_output_intermediate(self):
        # make sure the intermediate values are NOT freed
        i1 = trw.simple_layers.Input([None, 1], 'i1')
        i2 = trw.simple_layers.Input([None, 1], 'i2')
        intermediate = trw.simple_layers.Linear(i2, 3)
        n = trw.simple_layers.ConcatChannels([intermediate, i1, i2])
        n = trw.simple_layers.OutputEmbedding(n, output_name='concat')

        network = trw.simple_layers.compile_nn([n, intermediate])
        inputs = {
            'i1': torch.from_numpy(np.asarray([[1]], dtype=np.float32)),
            'i2': torch.from_numpy(np.asarray([[2]], dtype=np.float32))
        }
        outputs = network(inputs)

        # the order is important!
        assert len(outputs) == 2
        print(outputs['concat'].output)
        assert outputs[intermediate].shape == (1, 3)
        assert (outputs['concat'].output[:, 3:] == torch.from_numpy(np.asarray([[1, 2]], dtype=np.float32))).all()

    def test_compiled_multi_inputs_outputs(self):
        """
        Test the execution of a complex network and dependencies MUST be resolved in the correct order
        """
        input_1 = trw.simple_layers.Input(shape=[None, 1], feature_name='input_1')
        input_2 = trw.simple_layers.Input(shape=[None, 2], feature_name='input_2')
        input_3 = trw.simple_layers.Input(shape=[None, 3], feature_name='input_3')
        input_4 = trw.simple_layers.Input(shape=[None, 4], feature_name='input_4')
    
        tmp_1 = trw.simple_layers.ConcatChannels([input_1, input_2, input_4])
        output_1 = trw.simple_layers.OutputClassification(tmp_1, output_name='output_1', classes_name='output_1')
        
        last_2 = trw.simple_layers.ConcatChannels([tmp_1, input_3])
        output_2 = trw.simple_layers.OutputClassification(last_2, output_name='output_2', classes_name='output_2')
    
        net = trw.simple_layers.compile_nn([output_1, output_2])
        
        inputs = {
            'input_1': torch.zeros([5, 1]),
            'input_2': torch.zeros([5, 2]),
            'input_3': torch.zeros([5, 3]),
            'input_4': torch.zeros([5, 4]),
            'output_1': torch.zeros([5], dtype=torch.int64),
            'output_2': torch.zeros([5], dtype=torch.int64),
        }

        outputs = net(inputs)
        assert len(outputs) == 2
        assert outputs['output_1'].output.shape == torch.Size([5, 1 + 2 + 4])
        assert outputs['output_2'].output.shape == torch.Size([5, 1 + 2 + 4 + 3])

    def test_compiled_multi_inputs_outputs_2(self):
        """
        Test the execution of a complex network and dependencies MUST be resolved in the correct order
        """
        input_1 = trw.simple_layers.Input(shape=[None, 1], feature_name='input_1')
        input_2 = trw.simple_layers.Input(shape=[None, 2], feature_name='input_2')

        tmp_2 = trw.simple_layers.ReLU(input_2)
        tmp_3 = trw.simple_layers.ConcatChannels([input_1, tmp_2])

        output_1 = trw.simple_layers.OutputEmbedding(tmp_3, output_name='output_1')
        output_2 = trw.simple_layers.OutputEmbedding(tmp_2, output_name='output_2')
        
        net = trw.simple_layers.compile_nn([output_1, output_2])

        inputs = {
            'input_1': torch.zeros([5, 1]),
            'input_2': torch.zeros([5, 2]),
        }
        outputs = net(inputs)
        assert len(outputs) == 2
        
    def test_same_output_name_error(self):
        """
        We expect an error: 2 outputs can NOT have the same name!
        """
        input_1 = trw.simple_layers.Input(shape=[None, 1], feature_name='input_1')
        output_1 = trw.simple_layers.OutputEmbedding(input_1, output_name='output_1')
        output_2 = trw.simple_layers.OutputEmbedding(input_1, output_name='output_1')

        exception_raised = False
        try:
            trw.simple_layers.compile_nn([output_1, output_2])
        except:
            # we should get an exception
            exception_raised = True

        self.assertTrue(exception_raised)  # compile should have failed: 2 outputs with the same name!

    def test_reshape(self):
        input_1 = trw.simple_layers.Input(shape=[None, 32, 16], feature_name='input_1')
        o = trw.simple_layers.Reshape(input_1, [None, 32 * 16])
        net = trw.simple_layers.compile_nn([o])

        r = net({'input_1': torch.zeros((100, 32, 16))})
        assert r[o].shape == (100, 32 * 16)
