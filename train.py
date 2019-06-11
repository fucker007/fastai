from fastai.vision import *

path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
model = simple_cnn((3,16,16,2))
learn = Learner(data, model)
learn.fit(1)
learn.metrics=[accuracy]
learn.fit(1)

