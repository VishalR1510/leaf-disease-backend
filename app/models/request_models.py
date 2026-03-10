"""
Request-related type aliases and constants.

The actual upload comes as multipart/form-data (UploadFile),
so there is no JSON request body model.  This module holds
shared typing helpers and constants for the request layer.
"""

from typing import TypeAlias

# Raw bytes from the uploaded file
ImageBytes: TypeAlias = bytes
