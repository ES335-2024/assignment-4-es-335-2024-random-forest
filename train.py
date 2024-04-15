# Importing Files

import time
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,TensorBoard
import matplotlib.pyplot as plt


class VGG:
    def __init__(self, input_shape=(200,200,3), model_type='VGG3', n_class=1):
        self.input_shape = input_shape
        self.model_type = model_type
        self.n_class = n_class
        self.model = None
        self.train_acc = None
        self.train_loss = None
        self.train_time = 0
        self.test_acc = None
        self.n_parameters=0


    def build_model(self):
        model = Sequential()
        
        # VGG (1 Block)
        if self.model_type == 'VGG1':
            
            model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=self.input_shape))
            
            model.add(MaxPooling2D((2,2)))

        # VGG (3 Blocks)
        elif self.model_type == 'VGG3':
            
            model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=self.input_shape))
            
            model.add(MaxPooling2D((2,2)))
            
            model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
            
            model.add(MaxPooling2D((2,2)))
            
            model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
            
            model.add(MaxPooling2D((2,2)))
        
        # VGG16 (All layers fine tuning)
        elif self.model_type == 'VGG16_All':

            base_model = VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)

            for layer in base_model.layers:
                layer.trainable = True

            model.add(base_model)

        # VGG16 (Only MLP layers fine tuning)
        elif self.model_type == 'VGG16_MLP':
            base_model = VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)

            for layer in base_model.layers:
                layer.trainable = False

            model.add(base_model)

        # Invalid Model
        else:
            raise ValueError("Invalid model type. Please choose 'VGG1', 'VGG3', or 'VGG16'.")
        
        model.add(Flatten())

        model.add(Dense(256, activation='relu'))

        model.add(Dense(128, activation='relu'))

        model.add(Dense(self.n_class, activation='sigmoid'))

        self.model = model

    def train(self, filename, epochs, data_dir = 'dataset/', data_augmentation=False):

        self.build_model()
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # batches_per_epoch = X_train.shape[0] // batch_size
        # epochs = iterations // batches_per_epoch
        
        tensorboard = TensorBoard(log_dir='log/{}'.format(filename))
        checkpoint = ModelCheckpoint(filename + ".h5", monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')

        class LogBatchTensorBoard(TensorBoard):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def on_batch_end(self, batch, logs=None):
                logs = logs or {}
                for name, value in logs.items():
                    if name in ['batch', 'size']:
                        continue
                    summary = tf.Summary()
                    summary_value = summary.value.add()
                    summary_value.simple_value = value.item()
                    summary_value.tag = name
                    self.writer.add_summary(summary, batch)
                self.writer.flush()

        start_time = time.time()
        
        if not data_augmentation:
            # self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)
            Augmentor = ImageDataGenerator(rescale=1.0/255.0)
            train_itr = Augmentor.flow_from_directory(data_dir+'train/',class_mode='binary',batch_size=5,target_size=(200,200),seed = 42)
            test_itr = Augmentor.flow_from_directory(data_dir+'test/',class_mode='binary',batch_size=5,target_size=(200,200),shuffle=False)
            history = self.model.fit_generator(train_itr, steps_per_epoch=len(train_itr), validation_data=(test_itr), epochs=epochs, verbose=1, callbacks=[tensorboard,checkpoint,LogBatchTensorBoard(log_dir='log/{}'.format(filename))])

        else:
            # augmentor = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
            #                                horizontal_flip=True, vertical_flip=True)
            # augmentor.fit(X_train)
            # self.model.fit(augmentor.flow(X_train, Y_train, batch_size=batch_size), steps_per_epoch=batches_per_epoch,
            #                epochs=epochs, verbose=1)
            Augmentor = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rescale=1.0/255.0)
            Augmentor2 = ImageDataGenerator(rescale=1.0/255.0)
            
            train_itr = Augmentor.flow_from_directory(data_dir+'train/',class_mode='binary',batch_size=5,target_size=(200,200),seed = 42)
            test_itr = Augmentor.flow_from_directory(data_dir+'test/',class_mode='binary',batch_size=5,target_size=(200,200),shuffle=False)
            test_itr2 = Augmentor2.flow_from_directory(data_dir+'test/',class_mode='binary',batch_size=5,target_size=(200,200),shuffle=False)
            
            history = self.model.fit_generator(train_itr, steps_per_epoch=len(train_itr), validation_data=(test_itr), validation_steps=len(test_itr), epochs=epochs, verbose=1, callbacks=[tensorboard,checkpoint,LogBatchTensorBoard(log_dir='log/{}'.format(filename))])

        end_time = time.time()
        self.train_time = end_time - start_time
        
        self.train_acc = history.history['accuracy'][-1]
        
        self.train_loss = history.history['loss'][-1]
        
        self.n_parameters = self.model.count_params()

        test_loss,test_acc = self.model.evaluate_generator(test_itr,steps=len(test_itr),verbose=0)
        self.test_acc = test_acc
        self.summarize(history,filename)


        def images_and_predictions(model):
            all_filenames = []
            all_true_labels = []
            all_predicted_labels = []

            for i in range(len(test_itr)):
                test_imgs, test_labels = next(test_itr)
                preds = model.predict(test_imgs)
                preds = (preds > 0.5).astype(int)

                # Mapping the binary predictions to respective class_name
                class_names = ['Kangaroo', 'Sheep']
                predicted_classes = [class_names[int(prediction)] for prediction in preds]

                filenames = test_itr.filenames[i * test_itr.batch_size: (i + 1) * test_itr.batch_size]

                all_filenames.extend(filenames)
                all_true_labels.extend([class_names[int(label)] for label in test_labels])
                all_predicted_labels.extend(predicted_classes)

            # Print the filenames, true labels, and predicted labels for all batches
            for filename, true_label, predicted_label in zip(all_filenames, all_true_labels, all_predicted_labels):
                print(f"{filename} - True label: {true_label}, Predicted label: {predicted_label}")

            return test_imgs, test_labels, preds
        
        # Log test images and corresponding predictions to TensorBoard
        test_imgs, test_labels, preds = images_and_predictions(self.model)
        file_writer = tf.summary.create_file_writer(tensorboard.log_dir)
        with file_writer.as_default():
            tf.summary.image("Test Images", test_imgs,
                             max_outputs=len(test_imgs), step=0)
            tf.summary.text("Test Labels",
                            tf.strings.as_string(test_labels), step=0)
            tf.summary.text(
                "Predictions", tf.strings.as_string(preds), step=0)
        pass


    def training_time(self):
        return print("Training Time: ", self.train_time)


    def training_loss(self):
        return print("Training Loss: ", self.train_loss)


    def training_accuracy(self):
        return print("Training Accuracy: ", self.train_acc)


    def testing_accuracy(self):
        return print("Testing Accuracy: ", self.test_acc)


    def n_model_params(self):
        return print("Number of Model Parameters: ", self.n_parameters)


    def summarize(self,history,filename):
        plt.subplot(211)
        plt.title('Cross Entropy Loss')
        plt.plot(history.history['loss'], color='blue', label='train')
        plt.plot(history.history['val_loss'], color='orange', label='test')
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        # plot accuracy
        plt.subplot(212)
        plt.title('Classification Accuracy')
        plt.plot(history.history['accuracy'], color='blue', label='train')
        plt.plot(history.history['val_accuracy'], color='orange', label='test')
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.legend()
        # save plot to file
        plt.savefig(filename + '.png')
        plt.close()