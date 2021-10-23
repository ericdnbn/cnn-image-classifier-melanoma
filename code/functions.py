import os, shutil
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.image as mpimg
import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight


def visualize_images(images, name):
	fig, axs = plt.subplots(1, 5, figsize=(20, 5))

	for i, ax in enumerate(axs):
		ax.imshow(images[i], cmap='gray')
	
	fig.suptitle(f'{name}', fontsize=15, y=.8)
	fig.tight_layout;


def preprocessing_dense_data(train_directory=None,
                             test_directory=None,
                             ts_tuple=(256,256), 
                             color='rgb', 
                             batch_size=None,
                             process_test=False,
                             class_mode=None):

	'''
	Arguments:

	This function takes in a training and testing directory, 
	a tuple indicating how to resize the image, the color scale, 
	the number of images to pull from the directory, 
	and a boolean for process_test, which tells the function whether or not to create
	a test generator to pull images from the testing directory.

	Functionality:

	This function creates image generators with the flow_from_directory method called off 
	of them to pull images from their respective directories. 
	It then splits images pulled by each generator into variables and labels.
	The variables and labels are then separately reshaped to be in a single column.
	'''

	arg_dict = {'target_size':ts_tuple, 'color_mode':color, 'batch_size':batch_size, 'class_mode':class_mode}

	train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

	train_generator = train_datagen.flow_from_directory(train_directory, **arg_dict, subset='training')

	val_generator = train_datagen.flow_from_directory(train_directory, **arg_dict, subset='validation', shuffle=False)




	return train_generator, val_generator



	if process_test:

		train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

		train_generator = train_datagen.flow_from_directory(train_directory=train_directory, **arg_dict, subset='training')

		val_generator = train_datagen.flow_from_directory(train_directory=train_directory, **arg_dict, subset='validation', shuffle=False)

		test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(test_directory, **arg_dict, shuffle=False)


		return train_generator, val_generator, test_generator


def visualize_nn_test(history, model, train_generator, val_generator, test_generator, multi=None, labels=None):

	'''
	Arguments:

	This function takes in model history, the model itself, X_train, y_train, X_val, and y_val.

	Functionality:

	This function calculates accuracy, validation accuracy, loss, validation loss, 
	recall, and validation recall. It then plots these metrics, 
	evaluates the model on the training data, evaluates the model on the validation data,
	before finally plotting a confusion matrix showing the results 
	of using the model to predict the validation data.

	'''

	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	recall = history.history['recall']
	val_recall = history.history['val_recall']

	epochs = range(len(acc))
	plt.plot(epochs, acc, 'bo', label='Training accuracy')
	plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.figure()
	plt.plot(epochs, recall, 'bo', label='Training recall')
	plt.plot(epochs, val_recall, 'b', label='Validation recall')
	plt.title('Training and validation recall')
	plt.legend()
	plt.show()


	print('')
	print('Training Evaluation:')
	model.evaluate(train_generator)
	print('')
	print('Validation Evaluation:')
	model.evaluate(val_generator)
	print('')
	print('Testing Evaluation:')
	model.evaluate(test_generator)



	if multi:    

		preds = np.argmax(model.predict(test_generator), axis=-1)

		
		cm = confusion_matrix(test_generator.classes, preds)


		cmd = ConfusionMatrixDisplay(cm, display_labels=labels)

		print('')
		print('Test Confusion Matrix')
		print('')
		cmd.plot(values_format='d');


	else:

		preds = (model.predict_classes(test_generator) > 0.5).astype('int32')                


		cm = confusion_matrix(test_generator.classes, preds)


		cmd = ConfusionMatrixDisplay(cm, display_labels=labels)

		print('')
		print('Test Confusion Matrix')
		print('')
		cmd.plot(values_format='d');
    
    

def visualize_nn(history, model, train_generator, val_generator, multi=None, labels=None):

	'''
	Arguments:

	This function takes in model history, the model itself, X_train, y_train, X_val, and y_val.

	Functionality:

	This function calculates accuracy, validation accuracy, loss, validation loss, 
	recall, and validation recall. It then plots these metrics, 
	evaluates the model on the training data, evaluates the model on the validation data,
	before finally plotting a confusion matrix showing the results 
	of using the model to predict the validation data.

	'''

	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	recall = history.history['recall']
	val_recall = history.history['val_recall']

	epochs = range(len(acc))
	plt.plot(epochs, acc, 'bo', label='Training accuracy')
	plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.figure()
	plt.plot(epochs, recall, 'bo', label='Training recall')
	plt.plot(epochs, val_recall, 'b', label='Validation recall')
	plt.title('Training and validation recall')
	plt.legend()
	plt.show()


	print('')
	print('Training Evaluation:')
	model.evaluate(train_generator)
	print('')
	print('Validation Evaluation:')
	model.evaluate(val_generator)



	if multi:

		preds = np.argmax(model.predict(val_generator), axis=-1)

		
		cm = confusion_matrix(val_generator.classes, preds)


		cmd = ConfusionMatrixDisplay(cm, display_labels=labels)

		print('')
		print('Validation Confusion Matrix')
		print('')
		cmd.plot(values_format='d');


	else:

		preds = (model.predict_classes(val_generator) > 0.5).astype('int32')                


		cm = confusion_matrix(val_generator.classes, preds)


		cmd = ConfusionMatrixDisplay(cm, display_labels=labels)

		print('')
		print('Validation Confusion Matrix')
		print('')
		cmd.plot(values_format='d');




def nn_model(dense_filters, 
             train_directory=None,
             test_directory=None,
             ts_tuple=(256,256), 
             color='grayscale',
             batch_size=1000,
             process_test=False,
             class_mode='categorical',
             input_shape=(256,256,3),
             input_nodes=64,
             input_activation='relu',
             normal=False,
             dense_reg=False,
             dense_activation='relu',
             output_nodes=1,
             output_activation='sigmoid',
             l2_rate=0.01,
             optimizer='sgd',
             loss='binary_crossentropy',
             metrics=['accuracy', 'Recall'],
             epochs=50,
             visualize=True,
             multi=True,
             vis_labels=['Benign', 'Malignant', 'Unknown']):
	'''
	This function allows you to pull images from local directories using image data generators, 
	feed them into a customizable Convulational Neural Network, and visualize various
	metrics, as well as a confusion matrix.
	'''

	if not process_test:
		train_generator, val_generator = preprocessing_dense_data(train_directory=train_directory, batch_size=batch_size, class_mode=class_mode)
	else:
		train_generator, val_generator, test_generator = preprocessing_dense_data(train_directory=train_directory, test_directory=test_directory, batch_size=batch_size, class_mode=class_mode)


	nn_model = models.Sequential()

	nn_model.add(layers.Flatten(input_shape=input_shape))

	nn_model.add(layers.Dense(input_nodes, activation=input_activation))

	
	for ind, value in enumerate(dense_filters):

		if normal:

			if dense_reg:

				nn_model.add(layers.Dense(dense_filters[ind], use_bias=False, kernel_regularizer=l2(l2=l2_rate)))

				nn_model.add(layers.BatchNormalization())

				nn_model.add(Activation(dense_activation))

			else:

				nn_model.add(layers.Dense(dense_filters[ind], use_bias=False))

				nn_model.add(layers.BatchNormalization())

				nn_model.add(Activation(dense_activation))


		else:

			if dense_reg:

				nn_model.add(layers.Dense(dense_filters[ind], activation=dense_activation, kernel_regularizer=l2(l2=l2_rate)))

			else:

				nn_model.add(layers.Dense(dense_filters[ind], activation=dense_activation))


	nn_model.add(layers.Dense(output_nodes, activation=output_activation))

	print('ok')

	nn_model.compile(optimizer=optimizer,
                     loss=loss,
                     metrics=metrics)


	hist = nn_model.fit(train_generator, epochs=epochs, validation_data=(val_generator))

	if visualize:

		if process_test:

			visualize_nn_test(hist, nn_model, train_generator, val_generator, test_generator, multi=multi, labels=vis_labels)

		else:

			visualize_nn(hist, nn_model, train_generator, val_generator, multi=multi, labels=vis_labels)

	else:

		pass


	return nn_model



def cnn_preprocessing(train_directory=None,
                      test_directory=None,
                      ts_tuple=(256,256), 
                      color=None, 
                      batch_size=None,
                      class_mode=None,
                      process_test=False):


	'''
	Arguments:

	This function takes in a training and testing directory, 
	a tuple indicating how to resize the image, the color scale, 
	the number of images to pull from the directory, 
	and a boolean for process_test, which tells the function whether or not to create
	a test generator to pull images from the testing directory.

	Functionality:

	This function creates image generators with the flow_from_directory method called off 
	of them to pull images from their respective directories.
	'''

	arg_dict = {'target_size':ts_tuple, 
                'color_mode':color, 
                'batch_size':batch_size,
                'class_mode':class_mode}

	if process_test:

		train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

		train_generator = train_datagen.flow_from_directory(train_directory, **arg_dict, subset='training')

		val_generator = train_datagen.flow_from_directory(train_directory, **arg_dict, subset='validation', shuffle=False)

		test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(test_directory, **arg_dict, shuffle=False)

		return train_generator, val_generator, test_generator


	else:

		train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

		train_generator = train_datagen.flow_from_directory(train_directory, **arg_dict, subset='training') 

		val_generator = train_datagen.flow_from_directory(train_directory, **arg_dict, subset='validation', shuffle=False)

		return train_generator, val_generator
    


def visualize_cnn_test(history, model, train_generator, val_generator, test_generator, multi=None, labels=None):
   

	'''
	Arguments:

	This function takes in model history, the model itself, the train generator,
	the validation generator, and the test generator.

	Functionality:

	This function calculates accuracy, validation accuracy, loss, validation loss, 
	recall, and validation recall. It then plots these metrics, 
	evaluates the model on the training data, evaluates the model on the validation data, 
	evaluates the model on the test data,
	before finally plotting a confusion matrix showing the results 
	of using the model to predict the testing data.

	''' 

	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	recall = history.history['recall']
	val_recall = history.history['val_recall']


	epochs = range(len(acc))
	plt.plot(epochs, acc, 'bo', label='Training accuracy')
	plt.plot(epochs, val_acc, 'b', label='Test accuracy')
	plt.title('Training, and Test accuracy')
	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and Test loss')
	plt.legend()
	plt.figure()
	plt.plot(epochs, recall, 'bo', label='Training recall')
	plt.plot(epochs, val_recall, 'b', label='Validation recall')
	plt.title('Training and Test recall')
	plt.legend()
	plt.show()


	print('')
	print('Training Evaluation:')
	model.evaluate(train_generator)
	print('')
	print('Validation Evaluation:')
	model.evaluate(val_generator)
	print('')
	print('Testing Evaluation:')
	model.evaluate(test_generator)


	if multi:

		predictions = (model.predict_classes(test_generator))     

		preds = np.argmax(predictions)

		cm = confusion_matrix(test_generator.classes, preds)


		cmd = ConfusionMatrixDisplay(cm, display_labels=labels)

		print('')
		print('Test Confusion Matrix')
		print('')
		cmd.plot(values_format='d');


	else:

		preds = (model.predict_classes(test_generator) > 0.5).astype('int32')                


		cm = confusion_matrix(test_generator.classes, preds)


		cmd = ConfusionMatrixDisplay(cm, display_labels=labels)

		print('')
		print('Test Confusion Matrix')
		print('')
		cmd.plot(values_format='d');


def visualize_cnn(history, model, train_generator, val_generator, multi=None, labels=None):
    

	'''
	Arguments:

	This function takes in model history, the model itself, the train generator and
	the validation generator.

	Functionality:

	This function calculates accuracy, validation accuracy, loss, validation loss, 
	recall, and validation recall. It then plots these metrics, 
	evaluates the model on the training data, evaluates the model on the validation data, 
	before finally plotting a confusion matrix showing the results 
	of using the model to predict the validation data.

	'''
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	recall = history.history['recall']
	val_recall = history.history['val_recall']

	epochs = range(len(acc))
	plt.plot(epochs, acc, 'bo', label='Training accuracy')
	plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.figure()
	plt.plot(epochs, recall, 'bo', label='Training recall')
	plt.plot(epochs, val_recall, 'b', label='Validation recall')
	plt.title('Training and validation recall')
	plt.legend()
	plt.show();

	print('')
	print('Training Evaluation:')
	model.evaluate(train_generator)
	print('')
	print('Validation Evaluation:')
	model.evaluate(val_generator)


	if multi:
   

		preds = np.argmax(model.predict(val_generator), -1)

		
		cm = confusion_matrix(val_generator.classes, preds)


		cmd = ConfusionMatrixDisplay(cm, display_labels=labels)

		print('')
		print('Validation Confusion Matrix')
		print('')
		cmd.plot(values_format='d');


	else:

		preds = (model.predict_classes(val_generator) > 0.5).astype('int32')                


		cm = confusion_matrix(val_generator.classes, preds)


		cmd = ConfusionMatrixDisplay(cm, display_labels=labels)

		print('')
		print('Validation Confusion Matrix')
		print('')
		cmd.plot(values_format='d');


def cnn_model(cnn_filters,
              filters=[128],
              dense_filters=[512],
              kernel_size=(3,3),
              conv_activation='relu',
              input_shape=(256,256,1),
              pool_size=(2,2),
              five_by_five=False,
              five_filters=64,
              five_kernel_size=(5,5),
              five_activation='relu',
              l2_rate=0.01,
              conv_normal=False,
              conv_reg=False,
              conv_kernel_size=(5,5),
              conv_layer_activation='relu',
              dense_activation='relu',
              dense_reg=False,
              normal=False,
              output_nodes=1,
              output_activation='sigmoid',
              optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'Recall'],
              class_weights=False,
              steps_per_epoch=60,
              epochs=20,
              validation_steps=20,
              process_test=False,
              train_directory=None,
              test_directory=None,
              class_mode='categorical',
              ts_tuple=(256,256), 
              color='grayscale',
              batch_size=50,
              visualize=True,
              multi=True,
              vis_labels=['Benign', 'Malignant', 'Unknown']):
    
	'''
	This function allows you to pull images from local directories using image data generators, 
	feed them into a customizable Convulational Neural Network, and visualize various
	metrics, as well as a confusion matrix.
	'''

	if not process_test:
		train_generator, val_generator = cnn_preprocessing(train_directory=train_directory, batch_size=batch_size, color=color, process_test=process_test, class_mode=class_mode)

	else:
		train_generator, val_generator, test_generator = cnn_preprocessing(train_directory=train_directory, test_directory=test_directory, batch_size=batch_size, color=color, process_test=process_test, class_mode=class_mode)


	cnn_model = models.Sequential()


	for i, val in enumerate(cnn_filters):

		cnn_model.add(layers.Conv2D(cnn_filters[i], kernel_size=kernel_size, activation=conv_activation,
                                input_shape=input_shape))

		cnn_model.add(MaxPooling2D(pool_size))


	if five_by_five:

		cnn_model.add(Conv2D(five_filters, kernel_size=five_kernel_size, activation=five_activation, kernel_regularizer=l2(l2=l2_rate)))


	for i, val in enumerate(filters):

		if conv_normal:

			if conv_reg:

				cnn_model.add(layers.MaxPooling2D(pool_size))

				cnn_model.add(layers.Conv2D(filters[i], kernel_size=conv_kernel_size, use_bias=False, kernel_regularizer=l2(l2=l2_rate)))

				cnn_model.add(layers.BatchNormalization())

				cnn_model.add(layers.Activation(conv_layer_activation))

			else:

				cnn_model.add(layers.MaxPooling2D(pool_size))

				cnn_model.add(layers.Conv2D(filters[i], kernel_size=conv_kernel_size, use_bias=False))

				cnn_model.add(layers.BatchNormalization())

				cnn_model.add(layers.Activation(conv_layer_activation))


		else:

			if conv_reg:

				cnn_model.add(layers.MaxPooling2D(pool_size))

				cnn_model.add(layers.Conv2D(filters[i], kernel_size=conv_kernel_size, activation=conv_layer_activation, kernel_regularizer=l2(l2=l2_rate)))

			else:

				cnn_model.add(layers.MaxPooling2D(pool_size))

				cnn_model.add(layers.Conv2D(filters[i], kernel_size=conv_kernel_size, activation=conv_layer_activation))


	cnn_model.add(MaxPooling2D(pool_size))


	cnn_model.add(layers.Flatten())


	for ind, value in enumerate(dense_filters):

		if normal:

			if conv_reg:

				cnn_model.add(layers.Dense(dense_filters[ind], use_bias=False, kernel_regularizer=l2(l2=l2_rate)))

				cnn_model.add(layers.BatchNormalization())

				cnn_model.add(Activation(dense_activation))

			else:

				cnn_model.add(layers.Dense(dense_filters[ind], use_bias=False))

				cnn_model.add(layers.BatchNormalization())

				cnn_model.add(Activation(dense_activation))


		else:

			if conv_reg:

				cnn_model.add(layers.Dense(dense_filters[ind], activation=dense_activation, kernel_regularizer=l2(l2=l2_rate)))

			else:

				cnn_model.add(layers.Dense(dense_filters[ind], activation=dense_activation))


	cnn_model.add(layers.Dense(output_nodes, activation=output_activation))

	cnn_model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=metrics)


	print('ok')

	if class_weights:

		cnn_class_weights = get_class_weights()

		hist = cnn_model.fit(train_generator, 
                         	steps_per_epoch=steps_per_epoch, 
                         	epochs=epochs,  
                         	validation_data=(val_generator),
                         	validation_steps=validation_steps,
                         	class_weight=cnn_class_weights)

	else:

		hist = cnn_model.fit(train_generator,
							steps_per_epoch=steps_per_epoch, 
                         	epochs=epochs,  
                         	validation_data=(val_generator),
                         	validation_steps=validation_steps)


	if visualize:

		if process_test:

			visualize_cnn_test(hist, cnn_model, train_generator, val_generator, test_generator, multi=multi, labels=vis_labels)

		else:

			visualize_cnn(hist, cnn_model, train_generator, val_generator, multi=multi, labels=vis_labels)

	else:

		pass


	return cnn_model, hist  
