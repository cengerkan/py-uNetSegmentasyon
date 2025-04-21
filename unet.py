import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Encoder Bloğu
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        p = self.pool(x)
        return x, p

# Decoder Bloğu
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, skip_features):
        x = self.upconv(x)
        x = torch.cat((x, skip_features), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

# U-Net Modeli
class UNet(nn.Module):
    def __init__(self, num_classes=1):
        super(UNet, self).__init__()
        self.enc1 = EncoderBlock(3, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.dec1 = DecoderBlock(1024, 512)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)
        self.outputs = nn.Conv2d(64, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        b = self.bottleneck(p4)
        d1 = self.dec1(b, s4)
        d2 = self.dec2(d1, s3)
        d3 = self.dec3(d2, s2)
        d4 = self.dec4(d3, s1)
        return torch.sigmoid(self.outputs(d4))

# GPU Kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model çalıştırma cihazı: {device}")

# Modeli Tanımlama ve Cihaza Gönderme
model = UNet(num_classes=1).to(device)

# Optimizasyon ve Kayıp Fonksiyonu
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Veri Setini Yükleme Fonksiyonu
def load_data(data_dir, transform, target_size=(512, 512)):
    images = []
    masks = []
    classes = ['meningioma']
    for class_name in classes:
        image_dir = os.path.join(data_dir, class_name, 'images')
        mask_dir = os.path.join(data_dir, class_name, 'masks')
        image_filenames = sorted(os.listdir(image_dir))
        mask_filenames = sorted(os.listdir(mask_dir))
        for img_file, mask_file in zip(image_filenames, mask_filenames):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)
            img = Image.open(img_path).convert("RGB").resize(target_size)
            mask = Image.open(mask_path).convert("L").resize(target_size)
            img = transform(img)
            mask = transform(mask)
            images.append(img)
            masks.append(mask)
    return torch.stack(images), torch.stack(masks)

# Eğitme, Kaydetme ve Test Etme
if __name__ == '__main__':
    # Veri Seti Yolu
    data_dir = '/dataset'
    # Veri Setini Yükleme
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_images, train_masks = load_data(data_dir, transform)
    train_images, train_masks = train_images.to(device), train_masks.to(device)

    # Modeli Eğitme
    num_epochs = 10
    batch_size = 4
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, len(train_images), batch_size):
            batch_images = train_images[i:i+batch_size]
            batch_masks = train_masks[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_images)}")

    # Modeli Kaydetme
    torch.save(model.state_dict(), 'unet_model.pth')

    # Test için GPU Kontrolü ve Tahmin Etme
    def preprocess_test_image(image_path, transform, target_size=(512, 512)):
        img = Image.open(image_path).convert("RGB").resize(target_size)
        img = transform(img)
        return img.unsqueeze(0).to(device)

    test_image_path = 'test_image.png'
    test_image = preprocess_test_image(test_image_path, transform)
    model.eval()
    with torch.no_grad():
        predicted_mask = model(test_image).cpu().squeeze().numpy()

    # Tahmin Sonuçlarını Görselleştirme
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Test Görüntüsü')
    plt.imshow(test_image.cpu().squeeze().permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.title('Tahmin Edilen Maske')
    plt.imshow(predicted_mask, cmap='gray')
    plt.show()
