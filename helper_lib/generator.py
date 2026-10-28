import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def generate_samples(model, device, num_samples=10):
    model.to(device)
    model.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, 100, device=device)
        images = model(noise)
        grid = make_grid((images + 1) / 2, nrow=4)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid.cpu().permute(1, 2, 0))
    ax.axis('off')
    plt.show()
    return fig
