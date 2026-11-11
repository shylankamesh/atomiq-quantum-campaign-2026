"""Download the public organizer workbook and verify its recorded SHA-256."""
from pathlib import Path
import hashlib
from urllib.request import urlopen,Request

URL='https://drive.google.com/uc?export=download&id=1GV6QsrWGTyPStOLR_hjlGo2ZHq75C_4K'
SHA256='d2f8e63705af2883c7618c463a6ac2f351119d59a8869d2677f8825148e9a709'

if __name__=='__main__':
    path=Path(__file__).resolve().parent/'data/bank_campaign_participant_release.xlsx'
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==SHA256:
        print('Verified existing organizer workbook:',path)
    else:
        with urlopen(Request(URL,headers={'User-Agent':'Atomiq-reproducibility/1.0'}),timeout=60) as response:
            content=response.read()
        if hashlib.sha256(content).hexdigest()!=SHA256:
            raise ValueError('Organizer download differs from verified release; no file written. Download the participant workbook manually from the official resource folder and investigate the change.')
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        print('Downloaded and hash-verified organizer workbook:',path)
