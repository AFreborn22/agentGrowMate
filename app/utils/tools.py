import httpx
import os
from dotenv import load_dotenv
from typing import Optional
from datetime import date, datetime
from pydantic import EmailStr
from langchain_core.tools import tool

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8080/api/auth/update")

@tool
def updateData(
    nama: Optional[str] | None = None,
    tempat_lahir: Optional[str] | None = None ,
    tanggal_lahir: Optional[date] | None = None ,
    tanggal_kehamilan_pertama: Optional[date] | None = None ,
    pal: Optional[str] | None = None ,
    alamat: Optional[str] | None = None ,
    email: Optional[EmailStr] | None = None ,  
    berat_badan: Optional[float] | None = None ,
    tinggi_badan: Optional[float] | None = None ,
    lingkar_lengan_atas: Optional[float] | None = None 
) :
    """
    Alat ini digunakan untuk memperbarui data profil kesehatan ibu hamil. 
    Hanya panggil jika Bunda secara eksplisit meminta untuk mengubah 
    data profil seperti berat badan atau tinggi badan.

    Perhatian : 
    1. Jika tanggal lahir diubah, umur akan dihitung ulang secara otomatis. 
    2. Jika tanggal kelahiran pertama diubah, periode kehamilan akan dihitung berdasarkan tanggal kelahiran pertama.
    3. Jika user mengirimkan dalam format tanggal, bulan, tahun balik menjadi tahun, bulan tanggal
    Mohon pastikan bahwa perubahan data lain terkait diperbarui sesuai dengan perhitungan otomatis yang telah disediakan.
    """
    from app.utils.flow import getAgent
    agentInstance = getAgent()

    token = agentInstance.token

    if tanggal_kehamilan_pertama:
        tanggal_kehamilan_pertama = tanggal_kehamilan_pertama.isoformat()  
    if tanggal_lahir:
        tanggal_lahir = tanggal_lahir.isoformat() 

    updatePayload = {
        "nama" : nama,
        "tempat_lahir" : tempat_lahir,
        "tanggal_lahir" : tanggal_lahir,
        "tanggal_kehamilan_pertama" : tanggal_kehamilan_pertama,
        "pal" : pal,
        "alamat" : alamat,
        "email" : email,
        "berat_badan" : berat_badan,
        "tinggi_badan" : tinggi_badan,
        "lingkar_lengan_atas" : lingkar_lengan_atas
    }

    updatePayload ={k: v for k, v in updatePayload.items() if v is not None}
    print(updatePayload)

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