import httpx
import os
from typing import Optional
from langchain_core.tools import tool

API_URL = os.getenv("API_URL", "http://127.0.0.1:8080/api/auth/update")

@tool
def updateData(
    nama: Optional[str] = None,
    usia: Optional[int] = None,
    tempat_lahir: Optional[str] = None,
    tanggal_lahir: Optional[str] = None,
    alamat: Optional[str] = None,
    email: Optional[str] = None,
    berat_badan: Optional[float] = None,
    tinggi_badan: Optional[float] = None,
    lingkar_tangan: Optional[float] = None,
) :
    """
    Alat ini digunakan untuk memperbarui data profil kesehatan ibu hamil. 
    Hanya panggil jika Bunda secara eksplisit meminta untuk mengubah 
    data profil seperti berat badan atau tinggi badan.
    """
    from app.utils.flow import getAgent
    agentInstance = getAgent()

    token = agentInstance.token

    updatePayload = {
        "nama" : nama,
        "usia" : usia,
        "tempat_lahir" : tempat_lahir,
        "tanggal_lahir" : tanggal_lahir,
        "alamat" : alamat,
        "email" : email,
        "berat_badan" : berat_badan,
        "tinggi_badan" : tinggi_badan,
        "lingkar_tangan" : lingkar_tangan
    }

    updatePayload ={k: v for k, v in updatePayload.items() if v is not None}

    if not updatePayload :
        return "Gagal Memperbaharui data"
    
    try :
        headers = {
            "Authorization": f"Bearer {token}",   
            "Accept": "application/json",
        }
        response = httpx.put(f"{API_URL}", json=updatePayload, headers=headers)
        response.raise_for_status()

        return f"Data Anda telah berhasil diperbarui: {updatePayload}"
    except httpx.HTTPStatusError as e:
        return f"Gagal memanggil API Backend (Status {e.response.status_code}). Pastikan token valid."
    except Exception as e:
        return f"Terjadi error: {str(e)}"
    
UPDATE_TOOLS = [updateData]