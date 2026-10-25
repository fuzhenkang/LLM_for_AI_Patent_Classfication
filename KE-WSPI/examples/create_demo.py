import csv
from pathlib import Path
rows=[]
for i in range(250):
    ai=i%5==0
    text=("A neural network performs machine learning for anomaly detection." if ai
          else "A power transformer coil measures electrical voltage.")
    rows.append({"patent_id":f"SYN{i:05}","title":"Synthetic patent",
                 "abstract":text,"claims":text,
                 "cpc":"G06N3/08" if ai and i%3 else "G01R31/00","label":int(ai)})
path=Path(__file__).with_name("demo.csv")
with path.open("w",encoding="utf-8",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
print(path)
