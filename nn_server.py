# nn_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, ForwardRef
import uvicorn
from datetime import datetime

app = FastAPI(title="PageCraft NN Proxy Server", version="1.0.0")

ItemRef = ForwardRef('Item')

class Item(BaseModel):
    id: str
    x: float
    y: float
    width: float
    height: float
    rotation: float
    borderRadius: float
    color: str
    opacity: float
    padding: float
    paddingTop: float
    paddingRight: float
    paddingBottom: float
    paddingLeft: float
    margin: float
    marginTop: float
    marginRight: float
    marginBottom: float
    marginLeft: float
    boxShadowOffsetX: float
    boxShadowOffsetY: float
    boxShadowBlur: float
    boxShadowSpread: float
    boxShadowBaseColor: str
    boxShadowOpacity: float
    boxShadowUsesBoxOpacity: bool
    borderWidth: float
    borderColor: str
    borderStyle: str
    backgroundGradientEnabled: bool
    backgroundGradientStart: str
    backgroundGradientEnd: str
    backgroundGradientAngle: float
    filterBlur: float
    filterBrightness: float
    filterContrast: float
    zIndex: float
    # Optional extras
    backgroundImageEnabled: Optional[bool] = None
    backgroundImageSrc: Optional[str] = None
    backgroundImageSourceType: Optional[str] = None  # 'url' | 'gallery'
    backgroundImageGalleryId: Optional[str] = None
    backgroundImageFit: Optional[str] = None  # 'cover' | 'contain'
    backgroundImagePosition: Optional[str] = None  # 'center' | 'top' | etc.
    backgroundImageRepeat: Optional[str] = None  # 'no-repeat' | etc.
    children: Optional[List[ItemRef]] = None

Item.model_rebuild()

class GalleryImage(BaseModel):
    id: str
    name: str
    mimeType: str
    dataBase64: str
    width: Optional[int] = None
    height: Optional[int] = None

class SavedWork(BaseModel):
    version: int = 1
    savedAt: str
    itemsByResolution: Dict[str, List[Item]]
    gallery: List[GalleryImage]

class NNRequest(BaseModel):
    payload: SavedWork

class NNResponse(BaseModel):
    processedPayload: SavedWork

@app.post("/process", response_model=NNResponse)
async def process_nn(request: NNRequest):
    """
    Echoes back the same payload for now.
    TODO: Implement NN logic here (e.g., modify itemsByResolution with AI suggestions).
    """
    # For demo: Just return the input unchanged
    processed_payload = request.payload
    
    # Optional: Add a timestamp or log for debugging
    print(f"Received payload at {datetime.now().isoformat()}: {len(processed_payload.itemsByResolution)} resolutions")
    
    return NNResponse(processedPayload=processed_payload)

@app.get("/")
async def root():
    return {"message": "PageCraft NN Server is running! POST to /process for NN proxy."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)