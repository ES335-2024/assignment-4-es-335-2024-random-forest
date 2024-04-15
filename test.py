from train import *

mtype='VGG1'
fname='VGG1'
isAugmented=False

# Code to test the VGG network
vgg_model = VGG(model_type=mtype)
vgg_model.train(filename=fname,epochs=10,data_augmentation=isAugmented,data_dir='dataset/')
vgg_model.testing_accuracy()
vgg_model.training_accuracy()
vgg_model.training_loss()
vgg_model.n_model_params()
vgg_model.training_time()