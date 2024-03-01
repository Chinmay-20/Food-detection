from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

print("[INFO] loading network")

#model=InceptionResNetV2(weights="imagenet",include_top=True)
#model=VGG16(weights="imagenet",include_top=True)
model=MobileNetV2(weights="imagenet",include_top=False)
print("[INFO] showing layers")

for (i,layer) in enumerate(model.layers):
	print("[INFO] {}\t{}".format(i,layer.__class__.__name__))
	
