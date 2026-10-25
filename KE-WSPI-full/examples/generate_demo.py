import csv,random
from pathlib import Path
random.seed(42)
p=Path(__file__).with_name("patents.csv")
with p.open("w",encoding="utf-8",newline="") as f:
 w=csv.writer(f);w.writerow(["patent_id","title","abstract","claims","cpc","family_id","year","label"])
 for i in range(160):
  ai=(i%4==0);label=int(ai)
  desc="Neural network learns features and predicts anomalies in sensor readings." if ai else "Electrical power transformer winding controls voltage and magnetic flux."
  w.writerow([f"DEMO{i:05}","AI method" if ai else "Electrical device",desc,desc,
              "G06N3/08" if ai and i%3 else "H01F27/00",f"F{i:04}",2014+i%10,label])
print(p)
