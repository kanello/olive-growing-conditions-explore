# Olive Growing Conditions — Research Notes

> Phase 1 deliverable. Summarises environmental requirements for *Olea europaea* with cited thresholds.
> Each section feeds directly into feature selection and model sanity-checking in later phases.

---

## Temperature

| Variable | Threshold / Range | Notes | Source |
|---|---|---|---|
| Optimal mean annual temp | 15–20 °C | | |
| Absolute minimum (frost kill) | ~−10 °C | Varies by cultivar; Koroneiki tolerates down to −7 °C | |
| Chilling hours required | 200–300 h <7 °C | Needed for flower induction | |
| Max summer temp | up to 40 °C | Drought-adapted but heat causes fruit drop above ~38 °C | |

*Sources to consult: UC Davis Olive Center, FAO Ecocrop (olive entry)*

---

## Water / Precipitation

| Variable | Threshold / Range | Notes | Source |
|---|---|---|---|
| Annual precipitation | 400–800 mm | Tolerates 200 mm with irrigation | |
| Summer drought | Very high tolerance | Adapted to Mediterranean dry summer | |
| Waterlogging tolerance | Low | Requires well-drained soils | |

---

## Soil

| Variable | Preferred Range | Notes | Source |
|---|---|---|---|
| pH | 5.5–8.5 (optimum 6–8) | Tolerates calcareous soils well | |
| Texture | Sandy loam to clay loam | Needs drainage; pure clay problematic | |
| Drainage | Well-drained | Root rot risk in waterlogged soils | |
| Salinity | Moderate tolerance | | |

---

## Terrain

| Variable | Preferred Range | Notes | Source |
|---|---|---|---|
| Elevation | 0–600 m (Greece) | Some cultivation to 800 m in southern regions | |
| Slope | 0–30° | Terracing extends range on steeper ground | |
| Aspect | South-facing preferred | Maximises winter sun in northern latitudes | |

---

## Microclimate

- **Sea proximity**: coastal areas moderate winter frost risk; Aegean islands benefit from thermal buffering
- **Wind exposure**: persistent strong winds reduce yield; windbreaks common in exposed sites

---

## Key Discriminators vs. Other Crops

Based on Phase 3 feature engineering notes:
- **NDRE** (Sentinel-2 Red Edge): strong discriminator due to silver-grey leaf surface of olive
- **Summer precipitation deficit**: negative correlation — olives thrive where other crops cannot
- **Frost day count**: primary exclusion variable at northern range limits

---

## References

*(Fill in during Phase 1 research)*

- FAO Ecocrop: https://ecocrop.fao.org/
- UC Davis Olive Center publications
- Greek Ministry of Rural Development cultivar guides
- Academic: *Olea europaea* SDM literature (e.g., Guisan & Zimmermann 2000 framework applied to olives)
