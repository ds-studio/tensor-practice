from tensorflow.examples.tutorials.mnist import  input_data

mnist = input_data.read_data_sets("./mnist_data/", one_hot=True)

print("Training data size: ", mnist.train.num_examples)

print("validation data size: ", mnist.validation.num_examples)

print("test data size: ", mnist.test.num_examples)

print("Example training data: ", mnist.train.images[0])

print("Example trainning data label: ", mnist.train.labels[0])

xs, ys = mnist.train.next_batch(100)
print("X shape:", xs.shape)
print("Y shape:", ys.shape)