#Import neccessary library
import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.testing import test
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

#Custom datasets/dataloaders
class PennFudanDataset(Dataset):
    def __init__(self, root, img_size=None, transforms=None):
        self.img_size = img_size
        self.root = root
        self.transforms = transforms
        self.image_paths = list(sorted(os.listdir(os.path.join(root, 'PNGImages'))))
        self.mask_paths = list(sorted(os.listdir(os.path.join(root, 'PedMasks'))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'PNGImages', self.image_paths[idx])
        mask_path = os.path.join(self.root, 'PedMasks', self.mask_paths[idx])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
        image = Image.fromarray(image.astype('uint8')).convert('RGB')  # Convert to PIL Image

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask >= 0.5, 255, 0).astype('float32')
        mask = Image.fromarray(mask.astype('uint8')).convert('L')  # Convert to PIL Image

        # Apply transformations if provided
        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return image, mask

    def __len__(self):
        return len(self.image_paths)
    
'''
a) Cut the FudanPed dataset into an 80-10-10 train-val-test split.
b) Apply data augmentation to your dataset during training and
'''
transformations = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Adjust brightness, contrast, saturation, and hue
    transforms.Resize((128,128)),
    transforms.ToTensor()  # Convert PIL Image to PyTorch tensor
])
# Load the dataset and split into train, val, and test sets
dataset = PennFudanDataset(root='/content/drive/MyDrive/colab/CA4/archive', transforms=transformations)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
val_dataset, test_dataset = train_test_split(val_dataset, test_size=0.5, random_state=42)

'''
b) Show an example of your data augmentation in your report.
'''
image, true_mask = dataset[60]

# Convert to numpy arrays for visualization
image = np.array(image)
true_mask = np.array(true_mask)

# Plot the image and true
plt.imshow(example_image)
plt.title(f'Label: {"not_foul" if example_label == 0 else "foul"}')
plt.show()

'''
(c) Implement and train a CNN for binary segmentation on your train split. Describe
your network architecture2
, loss function, and any training hyper-parameters. You
may implement any architecture you’d like, but the implementation must be your
own code.
UNET implementation
'''

def double_convolution(in_channels, out_channels):
    """
    Dynamic double convolution block based on input and output channels.
    """
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return conv_op

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(UNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.UP = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # right path (downsampling)
        self.down1 = double_convolution(in_channels,16)
        self.down2 = double_convolution(16, 32)

        self.mid = double_convolution(32,32)

        # left path (upsampling)

        self.up1 = double_convolution(64,16)
        self.up2 = double_convolution(32,16)

        self.last = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
    def forward(self, x):
        down1 = self.down1(x)
        downSample1 = self.max_pool2d(down1)
        down2 = self.down2(downSample1)
        downSample2 = self.max_pool2d(down2)

        mid = self.mid(downSample2)

        upSample1 = self.UP(mid)
        up1 = torch.cat([upSample1, down2], dim=1)
        up1 = self.up1(up1)

        upSample2 = self.UP(up1)
        up2 = torch.cat([upSample2, down1], dim=1)
        up2 = self.up2(up2)

        # Concatenate skip connection 2 (from down1) with upSample2
        #final2d convo
        fi_nal = self.last(up2)

        # Apply sigmoid activation
        x = torch.sigmoid(fi_nal)

        return x
    

def dice_coefficient(pred, target):
    smooth = 1.  # Smoothing factor
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return  1- ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

%%capture
'''
(c) Implement and train a CNN for binary segmentation on your train split. Describe
your network architecture2
, loss function, and any training hyper-parameters. You
may implement any architecture you’d like, but the implementation must be your
own code.
Training Loop
'''

in_channels = 3  # RGB input channels
out_channels = 1  # 1 output channel for binary mask
num_epochs = 40
batch_size = 10  # 8 best?
learning_rate = 0.0001
gamma = 0.1

# Assuming train_dataloader, val_dataloader, and test_dataloader are defined appropriately
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the UNet model, BCE loss, and Adam optimizer
model = UNet(in_channels, out_channels)
criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for stability
# criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=gamma)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
train_loss_list = []
val_loss_list = []
val_dice_list = []
test_dice_list = []
model.train()
for epoch in range(num_epochs):
    total_loss = 0.0

    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

    # Print average training loss for the epoch
    avg_loss = total_loss / len(train_dataloader)
    train_loss_list.append(avg_loss)
    # print(f'Epoch {epoch + 1}/{num_epochs}, ')

    # Validation
    model.eval()

    with torch.no_grad():
        val_loss = []
        val_dices = []
        for val_inputs, val_targets in val_dataloader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

            # Forward pass
            val_outputs = model(val_inputs)

            # Calculate loss
            val_loss.append( criterion(val_outputs, val_targets).item())

            val_outputs = (val_outputs >= 0.5).float()  # Convert to binary predictions
            val_tgts =  torch.sigmoid(val_targets)
            val_dices.append( dice_coefficient(val_outputs, val_tgts).item())

        avg_val_loss = sum(val_loss) / len(val_dataloader)
        avg_dice_loss = sum(val_dices) / len(val_dataloader)
        val_loss_list.append(avg_val_loss)
        val_dice_list.append(avg_dice_loss)

    model.eval()

    with torch.no_grad():
        test_dices = []
        for test_inputs, test_targets in test_dataloader:
            test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)

            # Forward pass
            test_outputs = model(test_inputs)

            test_outputs = (test_outputs >= 0.5).float()  # Convert to binary predictions
            test_tgts = torch.sigmoid(test_targets)
            dice = dice_coefficient(test_outputs, test_tgts).item()
            test_dices.append(dice)

        avg_dice_loss1 = sum(test_dices) / len(test_dataloader)
        test_dice_list.append(avg_dice_loss1)

    scheduler.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, DICE: {avg_dice_loss:.4f}')

    model.train()

'''
(d) Report training loss, validation loss, and validation DICE curves. Comment on any
overfitting or underfitting observed.
'''
plt.figure(figsize=(10, 5))
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Curves')
plt.show()

'''
(d) Report training loss, validation loss, and validation DICE curves. Comment on any
overfitting or underfitting observed.

'''
# Plot validation Dice coefficient curve
plt.figure(figsize=(10, 5))
plt.plot(val_dice_list, label='Val Dice')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.legend()
plt.title('Validation Dice Coefficient Curve')
plt.show()
'''
(e) Report the average dice score over your test-set. You should be able to achieve a
score of around 0.7 or better.
'''
# Print average test Dice coefficient
avg_test_dice = sum(test_dice_list) / len(test_dice_list)
print(f'Average Test Dice Coefficient: {avg_test_dice:.4f}')

# Model + State_dict()
print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())


modelFileName = 'checkpoint.pth'
torch.save(model.state_dict(), modelFileName)

'''
(f) Show at least 3 example segmentations (i.e. show the RGB image, mask, and RGB
image × mask for 3 samples) from your training data and 3 from your testing data.
Comment on the generalization capabilities of your trained network.
'''


# Set the model to evaluation mode
model.eval()

# Move the model to GPU if available
model.to('cuda')

def visualize_segmentations(dataset, num_samples=3):
    for i in range(num_samples):
        # Get a random sample
        sample_idx = np.random.randint(len(dataset))
        sample_image, sample_mask = dataset[sample_idx]

        # Move input data to GPU
        sample_image = sample_image.to('cuda')

        # Convert to a batch of size 1 (add batch dimension)
        sample_image = sample_image.unsqueeze(0)

        # Forward pass to get the model's prediction
        with torch.no_grad():
            # Move input data to GPU
            predicted_mask = model(sample_image.to('cuda'))

        # Convert the predictions and ground truth back to numpy arrays
        sample_image_np = sample_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sample_mask_np = sample_mask.squeeze(0).cpu().numpy()
        predicted_mask_np = (predicted_mask.squeeze(0) >= 0.5).cpu().numpy()  # Binarize predictions with a threshold of 0.5

        # Plot the RGB image, ground truth mask, and RGB image overlaid with the mask
        plt.figure(figsize=(18, 6))

        # RGB Image
        plt.subplot(1, 3, 1)
        plt.imshow(sample_image_np)
        plt.title("RGB Image")

        # Ground Truth Mask
        plt.subplot(1, 3, 2)
        plt.imshow(sample_mask_np, cmap='gray')
        plt.title("Ground Truth Mask")

        # RGB Image overlaid with Predicted Mask
        plt.subplot(1, 3, 3)
        overlay = sample_image_np.copy()
        predicted_mask_np = predicted_mask_np[0, :, :]  # Assuming batch size is 1

        overlay[predicted_mask_np, :] = [255, 0, 0]  # Overlay in red where the mask is 1
        plt.imshow(overlay)
        plt.title("RGB Image with Predicted Mask Overlay")

        plt.show()

# Visualize examples from the training set
visualize_segmentations(train_dataset, num_samples=3)

# Visualize examples from the test set
visualize_segmentations(test_dataset, num_samples=3)

'''
(g) Show at least 1 example segmentation on an input image not from the FudanPed
dataset. Again, comment on the generalization capabilities of your network with
respect to this “out-of-distribution” image.

'''
from PIL import Image as PILImage

# Set the model to evaluation mode
model.eval()

# Move the model to GPU if available
model.to('cuda')

# Specify the path to the image file
image_path = '/content/drive/MyDrive/colab/CA4/fiq/tt.jpg'  # Replace with the actual file path

# Load the image using PIL
input_image = PILImage.open(image_path).convert('RGB')

# Apply any necessary preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize if needed
    transforms.ToTensor(),
    # Add more transformations as needed
])

input_image = transform(input_image)

# Move input data to GPU
input_image = input_image.unsqueeze(0).to('cuda')

# Forward pass to get the model's prediction
with torch.no_grad():
    predicted_mask = model(input_image)

# Convert the predictions back to a numpy array
predicted_mask_np = (predicted_mask.squeeze(0) >= 0.5).cpu().numpy()  # Binarize predictions with a threshold of 0.5

# Plot the original image and RGB image overlaid with the predicted mask
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(input_image.squeeze(0).permute(1, 2, 0).cpu().numpy())
plt.title("Original Image")

# RGB Image overlaid with Predicted Mask
plt.subplot(1, 2, 2)
overlay = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy().copy()
predicted_mask_np = predicted_mask_np[0, :, :]  # Assuming batch size is 1

overlay[predicted_mask_np, :] = [255, 0, 0]  # Overlay in red where the mask is 1
plt.imshow(overlay)
plt.title("RGB Image with Predicted Mask Overlay")

plt.show()





