Simplified neural network declaration
*************************************

Existing pain points with PyTorch
=================================

One of the pain point of the PyTorch API is that the user has to
specifiy the exact shape of the layer. This leads to a snowball effect
where changing one layer shape will affect the rest of the network.
Making even a simple tweak can be labor intensive. In effect, it is difficult 
to conveniently chain operations. The following is a very simple network
for the MNIST dataset to illustrate the problems:

.. testcode::

		class CNN(nn.Module):
			def __init__(self):
				super(CNN, self).__init__()
				
				# Pain point 1: we have to `declare` the layers
				# in a first stage. Then in `forward`, we need
				# to specify how to perform the calculation. This
				# is unnecessary and it complicates the code
				self.conv1 = nn.Sequential(
					nn.Conv2d(
						in_channels=1,              # Pain point 2: we have
						out_channels=16,            # to keep track of the input
						kernel_size=5,              # and output for each layer manually.
						stride=1,                   # This is really cumbersome!
						padding=2,                   
					),                               
					nn.ReLU(),                       
					nn.MaxPool2d(kernel_size=2),     
				)
				self.conv2 = nn.Sequential(          
					nn.Conv2d(16, 32, 5, 1, 2),      
					nn.ReLU(),                       
					nn.MaxPool2d(2),                
				)
				self.out = nn.Linear(32 * 7 * 7, 10)

		def forward(self, x):
			x = self.conv1(x)                   # Here we have to reuse the sub-networks
			x = self.conv2(x)                   # we declared earlier, making the code 
			x = x.view(x.size(0), -1)           # more verbose than necessary
			output = self.out(x)
			return output, x

Dealing with more complex networks (e.g., multiple inputs and multiple outputs),
render this situation even more cumbersome as we have to keep track of all the internal
parameters and how to execute them in the `forward` method.

Solution: :mod:`trw.simple_layers`
==================================

Using :mod:`trw.simple_layers`, we can solve these two pain points
and make the declaration of a CNN very simple. 
The following code is the equivalent of the previous network:

.. testcode::

	def create_net_and_execute(options, batch):
		# Declare the network: here we can easily chain the
		# operations and we avoid the snowball effect (i.e., the
		# the layer is not dependent on the previous layer parameters)
		n = trw.simple_layers.Input([None, 1, 28, 28], 'images')
		n = trw.simple_layers.Conv2d(n, out_channels=16, kernel_size=5)
		n = trw.simple_layers.ReLU(n)
		n = trw.simple_layers.MaxPool2d(n, 2)
		n = trw.simple_layers.Conv2d(n, out_channels=32, kernel_size=5)
		n = trw.simple_layers.ReLU(n)
		n = trw.simple_layers.MaxPool2d(n, 2)
		n = trw.simple_layers.ReLU(n)
		n = trw.simple_layers.Flatten(n)
		n = trw.simple_layers.Linear(n, 10)
		
		# Here we specify this node is a classification output and it requires
		# `targets` feature in the batch.
		n = trw.simple_layers.OutputClassification(n, output_name='softmax', classes_name='targets')
		
		# Optimmize the network so that the outputs can be
		# efficiently calculated
		compiled_nn = trw.simple_layers.compile_nn(output_nodes=[n])
		
		# Finally, calculate the outputs with some data
		outputs = compiled_nn(batch)
		
Now we can effortlessly chain sub-networks and easily add inputs and outputs as required. The
code is also simpler to read.