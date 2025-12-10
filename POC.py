import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load dataset
# -------------------------------

def load_cow_images(dataset_path):
    cows = {}
    # loop through folders in dataset
    for cow_id in sorted(os.listdir(dataset_path)):
        cow_dir = os.path.join(dataset_path, cow_id)

        if os.path.isdir(cow_dir):
            # collect image files
            images = [
                os.path.join(cow_dir, f)
                for f in os.listdir(cow_dir)
                if f.lower().endswith((".jpg", ".png"))
            ]
            if images:
                cows[cow_id] = images
    return cows


# -------------------------------
# 2. Load image
# -------------------------------

def load_face(image_path):
    img = cv2.imread(image_path)              # read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR → RGB
    return img


# -------------------------------
# 3. Create feature extractor
# -------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# load pretrained ResNet50
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# remove final classification layer → we get embeddings
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.to(device)
resnet.eval()  # inference mode

# preprocessing required for ResNet
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -------------------------------
# 4. Extract embedding
# -------------------------------

def get_embedding(img):
    img_t = preprocess(img).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():             # no training
        emb = resnet(img_t).squeeze().cpu().numpy()

    emb = emb / norm(emb)            # normalize vector
    return emb


# -------------------------------
# 5. Cosine similarity
# -------------------------------

def similarity(e1, e2):
    return dot(e1, e2)              # cosine = dot product of normalized vectors


# -------------------------------
# 6. Compute embeddings
# -------------------------------

dataset_path = "Cattely"
cows = load_cow_images(dataset_path)

print("Found cows:", list(cows.keys()))

cow_embeddings = {}

# loop through every image and compute embedding
for cow_id, image_paths in cows.items():
    cow_embeddings[cow_id] = []

    for img_path in image_paths:
        face = load_face(img_path)
        emb = get_embedding(face)
        cow_embeddings[cow_id].append(emb)

print("Embeddings computed!")


# -------------------------------
# 7. Show image
# -------------------------------

def show_image(image_path):
    img = load_face(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# compare first two cows
cowA, cowB = list(cows.keys())[:2]

print("\nComparing:")
print("Cow A =", cowA)
print("Cow B =", cowB)

embA = cow_embeddings[cowA][0]
embB = cow_embeddings[cowB][0]

sim = similarity(embA, embB)
print(f"Similarity between {cowA} and {cowB}: {sim:.3f}")

print("\nShowing Cow A face:")
show_image(cows[cowA][0])

print("\nShowing Cow B face:")
show_image(cows[cowB][0])


# -------------------------------
# Show two images
# -------------------------------

def show_side_by_side(path1, path2, title1="", title2=""):
    img1 = load_face(path1)
    img2 = load_face(path2)

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.title(title2)
    plt.axis("off")

    plt.show()


# -------------------------------
# 8. Identification test
# -------------------------------

def test_identification(cow_embeddings):
    cows_list = list(cow_embeddings.keys())

    # test a few cows
    for cow in cows_list[:5]:
        if len(cow_embeddings[cow]) < 2:
            continue

        same1 = cow_embeddings[cow][0]
        same2 = cow_embeddings[cow][1]

        # compare to next cow
        other = cows_list[(cows_list.index(cow) + 1) % len(cows_list)]
        diff = cow_embeddings[other][0]

        print("\n---", cow, "---")
        print("Same-cow similarity:", similarity(same1, same2))
        print(f"{cow} vs {other} similarity:", similarity(same1, diff))


print("\nRunning identification test...")
test_identification(cow_embeddings)


# -------------------------------
# Compare two specific cows
# -------------------------------

cow1 = "s1557"
cow2 = "s1607"

emb1 = cow_embeddings[cow1][0]
emb2 = cow_embeddings[cow2][0]

sim = similarity(emb1, emb2)

print(f"\nComparing {cow1} vs {cow2}")
print("Similarity =", sim)

show_side_by_side(
    cows[cow1][0],
    cows[cow2][0],
    title1=cow1,
    title2=f"{cow2} (sim={sim:.3f})"
)



