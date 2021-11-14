from pydantic import BaseModel

class ClassifierResponse(BaseModel):
    filename: str
    contentype: str
    predicted_class: str
