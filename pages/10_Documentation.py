# pages/01_📄_Documentation.py
import streamlit as st

st.title("Documentation")

st.markdown("""
## What this app does
This app loads a single **60-minute** resolution CSV and plots energy time series by **country/zone prefix**:
- Choose a prefix (e.g., `IT`, `DE_LU`, `SE_3`, `GB_GBN`…).
- Pick **power** series (left y-axis, MW) and **price** series (right y-axis).
- Filter by any **UTC** date window.
- Interact with a **Plotly** chart (click legend items to toggle series on/off).

## Data schema (minimum)
- **`utc_timestamp`** (UTC) — used as the x-axis and for time filtering.
- Other columns are named as `PREFIX_field_name`, e.g.:
  - `IT_load_actual_entsoe_transparency`
  - `DE_LU_price_day_ahead`
  - `SE_4_wind_onshore_generation_actual`

The app auto-discovers all available prefixes and fields from the CSV header.

## Power vs Price split
- **Price fields** = any field ending with `price_day_ahead` (EUR in most zones; GBP for `GB_GBN`).
- **Power fields** = all other numeric series under the chosen prefix (loads, solar, wind…).

## Residual Load (derived)
You can toggle a derived series:
residual_load = load_actual_entsoe_transparency
− (solar_generation_actual
+ wind_generation_actual
+ wind_onshore_generation_actual
+ wind_offshore_generation_actual)
- Any missing renewable component is treated as **0**.
- If **no** renewable fields exist for the prefix, residual = total load.

## File location
Default CSV path (editable on the main page):

## Notes
- The x-axis is always **UTC** (`utc_timestamp`).
- Use the legend to hide/show traces without reloading.
- If a chosen field doesn’t exist in the CSV for that prefix, it’s ignored gracefully.

## Next steps
- Show units per field in the legend (EUR vs GBP).
- Optional CET/CEST view using `cet_cest_timestamp`.
- Presets by country (e.g., “Load vs Forecast”).
- Lightweight data QA (NaNs, duplicates, gaps).
""")
