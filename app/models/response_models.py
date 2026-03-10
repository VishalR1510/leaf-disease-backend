"""
Pydantic response models for the Leaf Disease Detection API.
"""

from typing import Optional

from pydantic import BaseModel, Field


class LeafAnalysisData(BaseModel):
    """Structured analysis result combining Keras prediction + Groq reasoning."""

    plant_name: str = Field(..., description="Identified plant species")
    is_leaf: bool = Field(True, description="Whether the image contains a leaf")
    is_diseased: bool = Field(..., description="Whether the leaf shows signs of disease")
    disease_name: str = Field("Healthy", description="Name of the detected disease, or 'Healthy'")
    symptoms: str = Field("", description="Visible symptoms on the leaf")
    disease_description: str = Field("", description="Short description of the disease")
    possible_causes: str = Field("", description="Possible causes of the disease")
    preventive_measures: str = Field("", description="Preventive measures")
    treatment_suggestions: str = Field("", description="Treatment or management recommendations")


class ErrorDetail(BaseModel):
    """Structured error detail."""

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")


class APIResponse(BaseModel):
    """Standard API response envelope."""

    success: bool
    data: Optional[LeafAnalysisData] = None
    error: Optional[ErrorDetail] = None
