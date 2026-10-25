"""Only synthetic demonstrations. NEVER treat synthetic oracle as a genuine teacher."""
import pandas as pd
from pathlib import Path
path=Path(__file__).with_name("demo.csv")
df=pd.read_csv(path)
out=df[["patent_id"]].copy()
out["teacher_probability"]=df["label"].map({1:.9,0:.1})
out["teacher_confidence"]=.85
file=Path(__file__).with_name("teacher_demo.csv")
out.to_csv(file,index=False)
print(file)
