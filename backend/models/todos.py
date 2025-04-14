from pydantic import BaseModel

class Todo(BaseModel):
    name: str
    phone_number: str
    unique_id: int
    age: int
    