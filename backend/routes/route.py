from fastapi import APIRouter, HTTPException
from models.todos import Todo
from config.database import collection_name
from schema.schemas import list_serial
from bson import ObjectId
import requests
from pymongo.errors import PyMongoError, OperationFailure, ConnectionFailure

router = APIRouter()

#Get request for all patients
@router.get("/")
async def get_todos():
    try:
        todos = list_serial(collection_name.find())
        return todos
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail="Database error")

#Get request for single patient
@router.get("/patient/{patient_id}")
async def get_patient(patient_id: str):
    try:
        # First try to find by patient_id
        patient = collection_name.find_one({"unique_id": int(patient_id)})
        if patient is None:
            # If not found, try finding by _id
            patient = collection_name.find_one({"_id": ObjectId(patient_id)})
        
        if patient is None:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Convert ObjectId to string
        patient['_id'] = str(patient['_id'])
        return patient
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid patient ID format")
    except ConnectionFailure:
        raise HTTPException(status_code=503, detail="Database connection error")
    except OperationFailure as e:
        raise HTTPException(status_code=500, detail=f"Database operation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

#Post request
@router.post("/")
async def post_todo(todo: Todo):
    try:
        collection_name.insert_one(dict(todo))
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail="Database error")

#Put Request
@router.put("/{id}")
async def put_todo(id: str, todo: Todo):
    try:
        result = collection_name.find_one_and_update(
            {"_id": ObjectId(id)}, 
            {"$set": dict(todo)}
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Item not found")
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail="Database error")

#delete request
@router.delete("/{id}")
async def delete_node(id: str):
    try:
        result = collection_name.find_one_and_delete({"_id": ObjectId(id)})
        if result is None:
            raise HTTPException(status_code=404, detail="Item not found")
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail="Database error")

