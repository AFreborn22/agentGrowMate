import httpx
from langchain.agents import Tool

def updateUserData (userInput: str, nik: str):

    data = 'nama', 'usia', 'tempat_lahir', 'tanggal_lahir', 'alamat', 'berat_badan', 'tinggi_badan', 'lingkar_tangan'
    if data in userInput() :
        response = updateBackend(nik, userInput)

        return f"Berat badan Anda telah diperbarui menjadi {response}."
    return "Tidak ada perubahan yang diperlukan."

async def updateBackend(nik: str, userInput: float):
    url = "http://127.0.0.1:8080/api/auth/update"
    payload = {"nik": nik, "userInput": userInput}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
updateTool = Tool(
    name="Update Data",
    func=updateUserData,
    description="Fungsi untuk memperbarui data pengguna"
)