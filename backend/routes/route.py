from fastapi import APIRouter, HTTPException
from models.todos import Todo
from config.database import collection_name
from schema.schemas import list_serial
from bson import ObjectId
import requests


router = APIRouter()

#Get request for all patients
@router.get("/")
async def get_todos():
    todos = list_serial(collection_name.find())
    return todos

#Get request for single patient
@router.get("/{patient_id}")
async def get_patient(patient_id: str):
    try:
        patient = collection_name.find_one({"patient_id": patient_id})
        if patient is None:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Convert ObjectId to string
        patient['_id'] = str(patient['_id'])
        return patient
    
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


#Post request
@router.post("/")
async def post_todo(todo: Todo):
    collection_name.insert_one(dict(todo))

#Put Request
@router.put("/{id}")
async def put_todo(id: str, todo: Todo):
    collection_name.find_one_and_update({"_id": ObjectId(id)}, {"$set": dict(todo)} )

#delete request
@router.delete("/{id}")
async def delete_node(id: str):
    collection_name.find_one_and_delete({"_id": ObjectId(id)})


