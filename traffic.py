import torch
import cv2
import numpy as np
import os
import sys
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

EPOCHS = 50
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
BATCH_SIZE = 32

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = np.array(labels)
    
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size = TEST_SIZE
    )

    # Convert training and test data to torch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader for batching
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #mps is macOS GPU support
    print(f"Using device: {device}")

    # Get a compiled neural network
    model = get_model().to(device)

    # Set up optimizer and loss function for PyTorch
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    criterion = torch.nn.CrossEntropyLoss()

    print("Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

    print("Evaluating...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Final Test Accuracy: {100 * correct / total:.4f}")

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        torch.save(model.state_dict(), filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        if not os.path.isdir(category_dir):
            continue
        for filename in os.listdir(category_dir):
            filepath = os.path.join(category_dir, filename)
            # Read image using cv2
            img = cv2.imread(filepath)
            if img is None:
                continue
            # Resize image
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(category)
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(128 * 7 * 7, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, NUM_CATEGORIES),
    )

    print(model)

    return model


if __name__ == "__main__":
    main()
