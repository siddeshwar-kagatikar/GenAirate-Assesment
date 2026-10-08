## Data cleaning and integrity issues

| # | Problem | Evidence | Resolution |
|---:|---|---|---|
| 1 | **Duplicate claims** | 40 `-R` records duplicated corresponding base claims. | Removed the 40 `-R` records, leaving **2,000 unique claims**. |
| 2 | **Inconsistent `dateOfLoss` format** | `core` used ISO format such as `2024-09-26T00:00:00.000Z`; `legacy_bdx` used `DD-MM-YYYY`. | Parsed each source with its appropriate format and converted both to a common datetime representation. |
| 3 | **Payment dates before loss** | Six claims had `mostRecentPaymentDate = 1942-10-12` even though their losses occurred years later. | Treated those payment dates as invalid and set them to `NaN`. |
| 4 | **Construction date after loss** | Ten claims had `originalConstructionDate = 2025-06-01` while their losses occurred before 2025. | Set those construction dates to `NaN`. |
| 5 | **Invalid latitude values** | Three latitudes were outside the valid geographic range: `-93.5`, `-91.0`, and `-90.5`. | Set only the invalid latitude values to `NaN`. |
| 6 | **State-coordinate mismatch** | Several coordinates were geographically impossible for their stated state, such as Florida claims with latitude around `-82`. | Set only the inconsistent coordinate, latitude or longitude, to `NaN`; retained the claim. |
| 7 | **Monetary unit mismatch between sources** | `legacy_bdx` monetary values were approximately **1,000 times smaller** than equivalent `core` values. | Multiplied monetary fields in `legacy_bdx` by `1,000`. |
| 8 | **Zero property/replacement values** | 167 claims had zero property or replacement values, including claims with substantial damage or payment. | Treated zero `buildingPropertyValue` and `buildingReplacementCost` as missing rather than genuine `$0` values. |
| 9 | **Negative payment amounts** | One claim had negative building, contents, and net payment values. | Set negative payment values to `NaN`. |
| 10 | **Inconsistent flood-zone capitalization** | Values such as `ae`, `x`, `a`, and `ah` occurred alongside uppercase versions. | Standardized values with `strip().upper()`. |
| 11 | **Missing values** | Several fields had substantial missingness, including `floodWaterDuration` at approximately 79% and `floodEvent` at approximately 28%. | Did not delete rows; missing values were handled during ML preprocessing. |
| 12 | **High-cardinality ZIP code** | `reportedZipCode` had 1,047 unique values, with 752 occurring only once. | Dropped the field from the ML model because one-hot encoding would create sparse, poorly supported features. |
| 13 | **Strongly skewed monetary variables** | Many monetary variables had very large right tails. | Did not delete outliers; investigated extreme records and retained legitimate high-value claims. |
| 14 | **Negative water-depth values** | There were 110 negative values, including one `-99`. | Retained them because there was not enough schema or domain evidence to prove they were invalid. |
| 15 | **Extreme `floodWaterDuration`** | One claim had a duration of `195`, while most non-missing values were `1` or `2`. | Flagged it as suspicious but retained it because it could not be proven invalid. |
| 16 | **Damage greater than property or replacement value** | Six directly observable cases had building damage exceeding the property or replacement value. | Investigated the cases individually and retained them because the relationship was not sufficiently defined to classify them as data errors. |
| 17 | **Payment greater than coverage** | 21 claims had payment exceeding building coverage; one had an especially large gross payment. | Investigated the cases but retained them because payment fields appeared to have accounting or gross-payment semantics. |
| 18 | **Very low damage but high payment** | Nine claims had `buildingDamageAmount = $1` but payments greater than `$10,000`. | Flagged this as a semantic inconsistency but retained the claims because it could reflect claim or payment accounting behavior. |
| 19 | **Rare categorical values** | Rare flood-zone values such as `AHB`, `AOB`, and `D` were present. | Retained them because there was insufficient evidence that they were invalid. |
| 20 | **Categorical codes stored as numbers** | `occupancyType`, `elevatedBuildingIndicator`, and `primaryResidenceIndicator` were numeric dtypes but represented categories. | Treated them as categorical features during modeling rather than continuous numerical variables. |