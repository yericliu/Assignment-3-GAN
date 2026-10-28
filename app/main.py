from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.bigram_model import BigramModel
from classifier_api import router as classifier_router
from gan_api import router as gan_router

app = FastAPI(title="Text Generation, Image Classification and GAN")

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned "
    "and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str = Field(min_length=1)
    length: int = Field(ge=1, le=100)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    text = bigram_model.generate_text(
        request.start_word, request.length
    )
    return {"generated_text": text}


app.include_router(classifier_router)
app.include_router(gan_router)
