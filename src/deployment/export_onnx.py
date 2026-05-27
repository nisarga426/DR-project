import torch
import torch.nn as nn
import timm
import os

class DRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 5))
    def forward(self, x): return self.head(self.backbone(x))

def export():
    print("Loading trained model...")
    model = DRModel()
    model.load_state_dict(torch.load(
        os.path.expanduser('~/dr_project/best_model.pth'),
        map_location='cpu'))
    model.eval()
    print("Model loaded!")

    # Create dummy input (1 image, 3 channels, 300x300)
    dummy = torch.randn(1, 3, 300, 300)

    onnx_path = os.path.expanduser('~/dr_project/dr_model.onnx')

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=18,
        input_names=['retinal_image'],
        output_names=['grade_scores'],
        dynamic_axes={
            'retinal_image': {0: 'batch_size'},
            'grade_scores':  {0: 'batch_size'}
        }
    )
    print(f"ONNX model saved to: {onnx_path}")

    # Verify it works
    print("\nVerifying ONNX model...")
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(onnx_path)
    dummy_np = np.random.randn(1, 3, 300, 300).astype(np.float32)
    outputs = session.run(None, {'retinal_image': dummy_np})
    print(f"Output shape: {outputs[0].shape}")
    print("ONNX model verified and working!")

    # Compare sizes
    pth_size  = os.path.getsize(
        os.path.expanduser('~/dr_project/best_model.pth')) / 1e6
    onnx_size = os.path.getsize(onnx_path) / 1e6
    print(f"\nFile sizes:")
    print(f"  PyTorch (.pth):  {pth_size:.1f} MB")
    print(f"  ONNX (.onnx):    {onnx_size:.1f} MB")
    print(f"\nYour model is now deployable anywhere!")
    print(f"Share dr_model.onnx with any developer")
    print(f"They can run it without installing PyTorch!")

if __name__ == "__main__":
    export()
