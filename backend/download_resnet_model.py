# Download a CNN model for image recognition
# Download a ResNet model for image recognition
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Choose a ResNet model - ResNet-50 is a good balance of performance and size
model_name = "microsoft/resnet-50"  
save_path = os.path.join("models", "resnet-50")

# Download and save
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = ResNetForImageClassification.from_pretrained(model_name)

# Save locally
feature_extractor.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"ResNet model saved to {save_path}")