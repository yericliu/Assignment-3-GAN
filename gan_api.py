from io import BytesIO
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from helper_lib.model import Generator

router = APIRouter()
model = None


@router.get('/generate-gan')
def generate_gan():
    global model
    if model is None:
        model_path = Path(__file__).parent / 'gan_generator.pth'
        if not model_path.exists():
            raise HTTPException(status_code=503, detail='Train the GAN first.')
        model = Generator()
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval()

    with torch.no_grad():
        noise = torch.randn(16, 100)
        images = model(noise)
        grid = make_grid((images + 1) / 2, nrow=4)

    img = to_pil_image(grid)
    output = BytesIO()
    img.save(output, format='PNG')
    return Response(content=output.getvalue(), media_type='image/png')
