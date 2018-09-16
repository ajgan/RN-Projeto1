import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network
net = network.Network([784, 30, 10])

# Para testar com as diferentes classes
# newTest = []
# for i in range(len(test_data)):
#     if (test_data[i][-1] == 9):
#         newTest.append(test_data[i])
# net.SGD(training_data, 30, 10, 3.0, test_data=newTest)


net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
