import fastf1
import numpy as np 
import pandas as pd 
session = fastf1.get_session(2023, "Montreal", "Race")
session.load(telemetry=True, laps=True)

def dataframe_to_dict(df):
    result = {} 

    for col in df.columns:
        unique_values = df[col].unique() 
        result[col] = df[col].tolist() if len(unique_values) > 1 else unique_values[0] 
    return result 
# Load lap info and telemetry and telemetry as list of dicts
df_laps_telemetry, laps = [], []
for _, lap in session.laps.iterlaps():
    df_laps_telemetry.append(lap.get_telemetry())
    laps.append(lap.to_dict())
data = [dataframe_to_dict(tele) | lap for lap, tele in zip(laps, df_laps_telemetry)]
df = pd.DataFrame(data)

def interpolate_column(values):
    """Interpolates a list of values to a fixed length of 882."""
    x = np.linspace(0, 1, 882)
    xp = np.linspace(0, 1, len(values))
    return np.interp(x, xp, values)

df["Speed_emb"] = df["Speed"].apply(interpolate_column)
df["RPM_emb"] = df["RPM"].apply(interpolate_column)

import umap
import numpy as np
embeddings = np.stack(df["Speed_emb"].to_numpy())
reducer = umap.UMAP()
reduced_embedding = reducer.fit_transform(embeddings)
df["Speed_emb_umap"] = np.array(reduced_embedding).tolist()

import pandas as pd
from renumics import spotlight
spotlight.show(df, dtype={"Speed": spotlight.Sequence1D}, port=5436)