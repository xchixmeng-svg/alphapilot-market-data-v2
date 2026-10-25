import json
from pathlib import Path
import pandas as pd
p=Path('data/history/2020-2025/institutional_2020_2025.parquet')
x=pd.read_parquet(p)
res={'columns':list(map(str,x.columns)),'dtypes':{str(c):str(x[c].dtype) for c in x.columns},'rows':len(x),'head':x.head(3).astype(str).to_dict('records')}
Path('research/results/pivot_inst_schema_20261025').mkdir(parents=True,exist_ok=True)
Path('research/results/pivot_inst_schema_20261025/schema.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
print(json.dumps(res,ensure_ascii=False,indent=2))
