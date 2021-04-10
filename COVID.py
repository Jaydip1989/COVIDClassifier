


import numpy as np
import keras



img_rows,img_cols = 512,512
num_classes = 2
batch_size = 8




train_dir = "/Users/dipit/COVIDDataset/dataset/Train"
val_dir = "/Users/dipit/COVIDDataset/dataset/Val"
test_dir = '/Users/dipit/COVIDDataset/dataset/Prediction'


from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D,GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam


# In[5]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range = 25,
                                   zoom_range = 0.4,
                                   shear_range = 0.2,
                                   horizontal_flip = True)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)


# In[6]:


train_generator = train_datagen.flow_from_directory(
    directory = train_dir,
    target_size = (img_rows, img_cols),
    batch_size = batch_size,
    class_mode = "categorical",
    shuffle = True
)


# In[7]:


val_generator = val_datagen.flow_from_directory(
    directory = val_dir,
    target_size = (img_rows, img_cols),
    batch_size = batch_size,
    class_mode = "categorical",
    shuffle = False
)



# In[9]:


from keras.applications import InceptionV3
basemodel = InceptionV3(weights = 'imagenet',
                       input_shape = (img_rows,img_cols,3),
                       include_top = False)

for layer in basemodel.layers:
    layer.trainable = False

x = basemodel.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(2,activation="softmax")(x)
model = Model(inputs = basemodel.input, outputs = output)
model.summary()


# In[10]:


model.compile(loss = "categorical_crossentropy",
             optimizer = Adam(lr = 0.0001),
             metrics = ['acc'])


# In[11]:


history = model.fit(train_generator,
                   epochs = 5,
                   validation_data = val_generator)


# In[12]:


scores = model.evaluate(val_generator, verbose=1)
print("Loss: ",scores[0])
print("Accuracy: ",scores[1]*100)





pred = model.predict(val_generator, verbose=1)
y_pred = np.argmax(pred, axis=1)




from sklearn.metrics import classification_report, confusion_matrix



class_labels = val_generator.class_indices
class_labels = {v:k for k,v in class_labels.items()}
classes = list(class_labels.values())
print('Confusion Matrix')
print(confusion_matrix(val_generator.classes, y_pred))
print('Classification Report')
print(classification_report(val_generator.classes,y_pred,target_names=classes))

model.save('COVIDInception.h5')

import os
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import load_img,img_to_array
from os import listdir
from os.path import isfile, join
import re



def draw_test(name, pred,im, true_label):
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 160, 0, 0, 300, borderType=cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, "predicted-"+pred, (20, 60),cv2.FONT_HERSHEY_SIMPLEX, 1 ,(0,0,255),2)
    cv2.putText(expanded_image,"True-"+true_label,(20,120),cv2.FONT_HERSHEY_SIMPLEX,1 ,(0,255,0),2)
    cv2.imshow(name,expanded_image)

def getRandomImage(path, img_width, img_height):
    folders = list(filter(lambda x:os.path.isdir(os.path.join(path,x)),os.listdir(path)))
    random_directory = np.random.randint(0, len(folders))
    path_class = folders[random_directory]
    file_path = path + '/' + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0, len(file_names))
    image_name = file_names[random_file_index]
    final_path = file_path + '/' + image_name
    return image.load_img(final_path, target_size=(img_width,img_height)),final_path,path_class

img_width, img_height = 512, 512
files = []
predictions = []
true_labels = []

for i in range(0,10):
    path = '/Users/dipit/COVIDDataset/dataset/Val'
    img,final_path,true_label = getRandomImage(path, img_width, img_height)
    files.append(final_path)
    true_labels.append(true_label)
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x,axis=0)
    images = np.vstack([x])
    classes = np.argmax(model.predict(images,batch_size=8,verbose=0),axis=1)
    predictions.append(classes)

for i in range(0,len(files)):
    image = cv2.imread((files[i]))
    draw_test("Prediction", class_labels[predictions[i][0]], image, true_labels[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()




