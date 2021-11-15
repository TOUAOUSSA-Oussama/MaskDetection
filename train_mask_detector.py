# Import the necessary packages :
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# Defintion des hyperparametres :
learning_rate = 1e-4
epochs = 20
batch_size = 32

# path de la base de donnes :
directory = r"C:\Users\admin\OneDrive\Bureau\Etudes\Projets\Mask Detection\dataset"
categories = ["with_mask", "without_mask"]

# Laoding data :
data = []
labels = []

for category in categories:
    path = os.path.join(directory, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image) #pour adapter la dimension au model de keras

        data.append(image)
        labels.append(category)

# convertir les labels to one-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels) # c'est soit 0 ou 1

# diviser les donnees en ceux pour l'entrainement (80%) et ceux pour test (20%)
data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, 
                                    stratify=labels, random_state=42)

# Data augmentation :
aug = ImageDataGenerator(
    rotation_range = 20,
    zoom_range = 0.15,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.15,
    horizontal_flip = True,
    fill_mode = "nearest"
)

# define the model :
baseModel = MobileNetV2(weights="imagenet", include_top=False,
            input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will not b eupdated during
# the first training process
for layer in baseModel.layers:
    layer.trainable = False

# compile the model : defintion de l'optimisateur et du loss
optim = Adam(lr = learning_rate, decay=learning_rate/epochs)
model.compile(loss="binary_crossentropy", optimizer=optim, metrics=["accuracy"])

# train the model :
print("Model training :")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=batch_size),
    steps_per_epoch = len(trainX)//batch_size,
    validation_data = (testX, testY),
    validation_steps = len(testX)//batch_size,
    epochs = epochs
)

# Afficher loss et accuracy lors de l'entrainement :
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

# Prediction :
print("Prediction part :")
predIndice = model.predict(testX, batch_size=batch_size)
predIndice = np.argmax(predIndice, axis=1) # pour avoir l'indice qui a la proba maximal 

# Show metrics :
print(classification_report(testY.argmax(axis=1), predIndice, target_names=lb.classes_))

# sauvegarder le modele entraine :
model.save("mask_detector.model", save_format="h5")