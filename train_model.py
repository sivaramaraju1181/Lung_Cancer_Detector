from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

data = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train = data.flow_from_directory(
    "dataset",
    target_size=(224,224),
    batch_size=16,
    class_mode="binary",
    subset="training"
)

val = data.flow_from_directory(
    "dataset",
    target_size=(224,224),
    batch_size=16,
    class_mode="binary",
    subset="validation"
)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(train, validation_data=val, epochs=5)

model.save("model.h5")

print("Model trained successfully!")