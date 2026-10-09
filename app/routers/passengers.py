"""Package 1: Passenger & Travel Documents."""
import sqlite3
from typing import List

from fastapi import APIRouter, Depends

from .. import real_face, services
from ..database import get_db
from ..schemas import (
    PassengerDetail, PassengerSummary, ScanRequest, SessionOut,
    UploadPassportRequest, UploadPassportResponse,
)

router = APIRouter(prefix="/api/passengers", tags=["1. Passenger & Travel Documents"])


@router.get("", response_model=List[PassengerSummary], summary="List enrolled passengers")
def list_passengers(conn: sqlite3.Connection = Depends(get_db)):
    return services.list_passengers(conn)


@router.get("/{passenger_id}", response_model=PassengerDetail,
            summary="Passenger with passport, boarding passes and template metadata")
def get_passenger(passenger_id: str, conn: sqlite3.Connection = Depends(get_db)):
    return services.get_passenger_detail(conn, passenger_id.strip().upper())


@router.post("/scan", response_model=SessionOut, status_code=201,
             summary="Step 1: scan a passport and/or boarding pass and verify the documents")
def scan_documents(payload: ScanRequest, conn: sqlite3.Connection = Depends(get_db)):
    return services.start_session(
        conn,
        passport_number=payload.passport_number,
        booking_id=payload.booking_id,
        kiosk_id=payload.kiosk_id,
        passport_image_b64=payload.passport_image_b64,
    )


@router.post("/extract-passport-photo", response_model=UploadPassportResponse,
             summary="Extract face portrait from uploaded passport image")
def extract_passport_photo(payload: UploadPassportRequest, conn: sqlite3.Connection = Depends(get_db)):
    doc_res = real_face.extract_face_from_document(payload.passport_image_b64)
    session = services.start_session(
        conn,
        passport_number=payload.passport_number,
        booking_id=payload.booking_id,
        kiosk_id=payload.kiosk_id,
        passport_image_b64=payload.passport_image_b64,
    )
    return UploadPassportResponse(
        session=session,
        face_detected=doc_res.get("face_detected", False),
        face_thumbnail_b64=doc_res.get("face_thumbnail_b64"),
        message=doc_res.get("message", "Passport processed"),
    )

