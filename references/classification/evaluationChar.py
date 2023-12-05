import torch
from doctr.models import classification
from PIL import Image
from doctr import transforms as T

from doctr.io.image import tensor_from_pil


from torchvision.transforms.v2 import (
    Compose,
    GaussianBlur,
    InterpolationMode,
    Normalize,
    RandomGrayscale,
    RandomPerspective,
    RandomPhotometricDistort,
    RandomRotation,
    ToTensor,
)

import os


test_image_folder = "/home/navaneeth/Z_RENAISSANCE/diacriticsOCR/doctr/output/val/"
test_images = os.listdir(test_image_folder)
test_images.sort()

checkpoint_path = (
    "/home/navaneeth/Z_RENAISSANCE/diacriticsOCR/doctr/diacriticsOCR_temp_backup.pt"
)

model = classification.__dict__["vgg16_bn_r"](
    pretrained=False, num_classes=8, classes=["A", "E", "K", "P", "H", "R", "F", "B"]
)
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint)

img_transforms = Compose(
    [
        T.Resize((32, 32)),
        # Ensure we have a 90% split of white-background images
        T.RandomApply(T.ColorInversion(), 0.9),
    ]
)

predictions = []

i = 0
for test_image in test_images:
    image_path = os.path.join(test_image_folder, test_image)
    image = Image.open(image_path).convert("RGB")
    print(image_path)

    # Preprocess the image (normalize, resize, etc.)
    image = tensor_from_pil(image)
    # print("Image type", type(image))
    image = img_transforms(image)
    # print("Image type 2", type(image))

    """resize_transform = T.Resize((32, 32))
    resized_image = resize_transform(image)

    # Random Apply Color Inversion with 90% probability
    color_inversion_transform = T.RandomApply(T.ColorInversion(), p=0.9)
    color_inverted_image = color_inversion_transform(color_inversion_transform)"""

    # Normalize
    normalize_transform = Normalize(
        mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301)
    )
    normalized_tensor = normalize_transform(image)

    # Now normalized_tensor is a PyTorch tensor
    # print(normalized_tensor)
    normalized_tensor = normalized_tensor.unsqueeze(0)

    # Perform a forward pass through the model
    out = model(normalized_tensor)

    # Print the output tensor
    # print(out)

    # probabilities = torch.nn.functional.softmax(out[0], dim=0)
    # print(probabilities)

    # Get the index of the predicted class
    # predicted_class = torch.argmax(probabilities).item()

    # Print the predicted class and corresponding probability
    # print("\n" + image_path)
    # print("Predicted Class softmax:", predicted_class)

    predicted_class = out.argmax(dim=1)
    print("Predicted Class argmax:", predicted_class.item())

    # i = i + 1
    # print("Probability:", probabilities[predicted_class].item())
