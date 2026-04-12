# Dallas Building Permits — Initial Data Analysis & Observations

**Project:** Predict estimated building permit price (declared construction value) for new buildings  
**Role:** Senior Data Scientist — US Government Urban Planning  
**Date:** 2026-04-11  
**Phase:** Pre-Cleaning Exploratory Analysis

---

## 1. Project Context & Document Inventory

We have three source documents and one data file:

| File | Purpose | Size |
|------|---------|------|
| [Building-Permits.csv](file:///c:/Users/saari/Downloads/ASDS-6302-Final-Project/Building-Permits.csv) | Raw dataset — 126,840 permit records from Dallas OpenData | ~31 MB |
| [Dallas_BuildingPermits_Execution_Guide.md](file:///c:/Users/saari/Downloads/ASDS-6302-Final-Project/Dallas_BuildingPermits_Execution_Guide.md) | 16-step end-to-end project guide (EDA → ML → Flask API → n8n Agent) | 17 KB |
| [Claude.md](file:///c:/Users/saari/Downloads/ASDS-6302-Final-Project/Claude.md) | MVP module spec — data cleaning, feature engineering, model training | 2 KB |

> [!IMPORTANT]
> **Critical discrepancy detected.** The Execution Guide references columns by their Socrata API names (e.g., `declared_value`, `permit_subtype`, `issue_date`, `council_district`), but the actual CSV uses a different schema (e.g., `Value`, `Issued Date`, `Zip Code`). The Claude.md MVP spec uses the CSV column names. **All downstream work must use the actual CSV column names and map guide instructions accordingly.**

### Column Name Mapping (Guide → Actual CSV)

| Execution Guide Name | Actual CSV Column | Notes |
|---|---|---|
| `permit_number` | `Permit Number` | ✅ Direct map |
| `permit_type` | `Permit Type` | ✅ But format is composite (see §3) |
| `permit_subtype` | — (embedded in `Permit Type`) | ⚠️ Must be **parsed** out of `Permit Type` |
| `issue_date` | `Issued Date` | ✅ 2-digit year format (`MM/DD/YY`) |
| `council_district` | — **MISSING** | ❌ Not in the CSV at all |
| `contractor_name` | `Contractor` | ⚠️ Contains name + full address + phone |
| `declared_value` | `Value` | ✅ Comma-formatted string |
| `square_footage` | `Area` | ✅ Comma-formatted string |
| `work_description` | `Work Description` | ✅ Direct map |
| `occupancy_type` | `Land Use` | ✅ Semantic equivalent |
| `address` | `Street Address` | ✅ Direct map |
| `zip_code` | `Zip Code` | ✅ Direct map |
| — | `Mapsco` | 🆕 Extra column — Dallas grid reference system |

> [!WARNING]
> **`council_district` is referenced 8+ times in the Execution Guide** (imputation logic, feature engineering, one-hot encoding, Flask API input) **but does not exist in the CSV**. We have two options:
> 1. **Drop it** from all downstream steps — simplest, no data loss.
> 2. **Derive it** from `Zip Code` or `Mapsco` using a Dallas council district lookup — adds accuracy but requires external data.
> 
> **Recommendation:** Drop `council_district` for the MVP. It can be reverse-geocoded from `Street Address` in a future iteration if needed.

---

## 2. Dataset Overview

| Metric | Value |
|--------|-------|
| **Total rows** | 126,840 |
| **Total columns** | 11 |
| **Date range** | Jan 2, 2018 → Aug 29, 2020 (~2.7 years) |
| **Unique permit numbers** | 126,840 (no duplicates — clean primary key) |
| **Unique permit types** | 137 |
| **Unique contractors** | 14,031 |
| **Unique zip codes** | 71 |
| **Unique land uses** | 189 |

### Permits by Year

| Year | Count | % of Total |
|------|-------|------------|
| 2018 | 47,344 | 37.3% |
| 2019 | 52,037 | 41.0% |
| 2020 | 27,459 | 21.7% |

> [!NOTE]
> **2020 drop is expected** — the data ends Aug 29, 2020, so it captures only ~8 months. Additionally, COVID-19 lockdowns (March 2020) likely suppressed permit activity. This temporal trend is important context for the chronological train/val/test split.

---

## 3. Column-Level Profiling

### 3.1 Target Variable: `Value` (a.k.a. Declared Construction Value)

| Statistic | Value |
|-----------|-------|
| Data type in CSV | String with commas (e.g., `"665,000"`) |
| Min | 0 |
| Max | 64,565,291 |
| Mean | $87,655 |
| Median | $2,800 |
| **Zeros** | **30,054 (23.69%)** |
| Negatives | 0 |

**Percentile Distribution:**

| Percentile | Value ($) |
|------------|-----------|
| 1st | 0 |
| 5th | 0 |
| 10th | 0 |
| 25th | 300 |
| 50th (median) | 2,800 |
| 75th | 11,566 |
| 90th | 75,719 |
| 95th | 260,000 |
| 99th | 2,000,000 |

> [!IMPORTANT]
> **Key findings on the target variable:**
> 1. **Extremely right-skewed** — mean ($87.6K) is 31× the median ($2.8K). Log-transformation is mandatory for regression.
> 2. **23.69% of records have Value = 0** — nearly 1 in 4 permits. These are not "free" projects; they represent permit types that don't require a declared value (e.g., Electrical Signs = 100% zero, Demolitions = 100% zero, Barricades = 99% zero). **These must be removed** — you cannot log-transform zero, and they are not valid regression targets.
> 3. **Heavy tail** — the 99th percentile ($2M) is 23× the 90th percentile ($75.7K). The guide's recommendation to cap at the 99th percentile is well-justified.

**Top Permit Types Where Value = 0:**

| Permit Type | Zero-Value Count | Total Count | % Zero |
|---|---|---|---|
| Electrical Sign New Construction | 5,171 | 5,171 | **100%** |
| Building (BU) Single Family Renovation | 3,610 | 8,074 | 44.7% |
| Demolition Permit SFD/Duplex | 2,233 | 2,233 | **100%** |
| Electrical (EW) Commercial Alteration | 2,058 | 2,136 | 96.3% |
| Building (BU) Single Family New Construction | 1,915 | 5,131 | 37.3% |

> [!NOTE]
> **Decision rationale for zero-value removal:** Sign permits, demolitions, and barricade permits are administrative permits with no meaningful "construction value" — the city sets the fee regardless of project cost. Including them would create a bimodal target distribution that poisons the regression model. We remove all `Value == 0` rows, which leaves us with ~96,786 records — still a large, healthy training set.

### 3.2 `Area` (Square Footage)

| Statistic | Value |
|-----------|-------|
| Min | 0 |
| Max | 2,478,361 |
| Mean | 1,694 |
| Median | **0** |
| **Zeros** | **77,165 (60.84%)** |

> [!WARNING]
> **60.8% of Area values are zero.** This is a massive missing-data problem disguised as valid data. Zero square footage makes physical sense only for non-building permits (electrical, plumbing, sign, barricade, etc.) where no new floor area is created. For building/renovation permits, zero likely means "not reported."
>
> **Decision:** After filtering out Value-zero rows, we will re-check the Area zero rate. For the remaining records, we will impute Area zeros using **median Area grouped by the parsed Permit Category + Work Type** (e.g., median area for "Building / Single Family / New Construction" is meaningful; median area for "Plumbing / Single Family / Alteration" is correctly 0).

**Correlation with Value:** Pearson r = **0.486** (for records where both Value > 0 and Area > 0). This is a moderately strong positive relationship — larger buildings cost more, as expected. This is a critical feature.

### 3.3 `Permit Type` — Composite Field

The `Permit Type` field encodes **four pieces of information** in a single string:

```
Building (BU) Multi Family  New Construction
───────── ──── ──────────── ────────────────
Category  Code  Sub-Category   Work Type
```

**Parsed components:**

| Component | Examples | Unique Values |
|-----------|----------|---------------|
| **Category** | Building, Electrical, Plumbing, Mechanical, Fence, Swimming Pool | ~20 |
| **Code** | BU, EL, PL, ME, FE, SW, LS, SE, FA, BA | ~15 |
| **Sub-Category** | Single Family, Multi Family, Commercial | 3 main |
| **Work Type** | New Construction, Alteration, Renovation, Addition | ~5 |

> [!NOTE]
> **Decision:** Parse `Permit Type` into four separate features: `permit_category`, `permit_code`, `sub_category`, and `work_type`. This gives the ML model access to individual dimensions rather than forcing it to learn from 137 one-hot columns.
>
> **Edge case:** ~5% of permit types don't follow the standard pattern (e.g., "Electrical Sign New Construction", "Demolition Permit SFD/Duplex"). These will need a fallback parser or manual mapping.

**Top 5 Permit Categories by Volume:**

| Category | Count | Mean Value ($) |
|---|---|---|
| Plumbing — Single Family — Alteration | 21,398 | $2,716 |
| Building — Single Family — Alteration | 10,316 | $13,317 |
| Mechanical — Single Family — Alteration | 10,205 | $9,955 |
| Building — Commercial — Renovation | 9,116 | **$253,475** |
| Electrical — Single Family — Alteration | 8,663 | $3,204 |

> The mean value of Commercial Renovation permits ($253K) is **93× higher** than Single Family Plumbing ($2.7K). The sub-category (Commercial vs. Single Family) is a **very high-signal feature**.

### 3.4 `Issued Date`

- **Format:** `MM/DD/YY` (2-digit year) — e.g., `03/13/20`
- **Range:** 2018-01-02 to 2020-08-29
- **No nulls, no parse errors** — clean temporal column
- The Execution Guide calls this `issue_date`; the Claude.md spec also references an `Application Date` that **does not exist** in this CSV.

> [!WARNING]
> The Claude.md MVP spec defines a target variable `Processing_Time = Issue Date - Application Date`. But **we only have `Issued Date`**, not `Application Date`. We cannot compute processing time. The project must use **`Value` (construction value) as the prediction target**, consistent with the Execution Guide.

### 3.5 `Contractor`

- **14,031 unique values** — high cardinality
- Format: `COMPANY_NAME ADDRESS, CITY, STATE ZIP (PHONE)`
- Contains embedded contractor city (can be extracted) — e.g., "Dallas", "Garland", "Plano", "Irving"
- Only **3 null values** (negligible)

> **Decision:** Parse the `Contractor` field to extract:
> 1. `contractor_name` — for computing `contractor_permit_count` and `is_repeat_contractor`
> 2. `contractor_city` — for frequency encoding (as per the Execution Guide)
>
> **Why:** Raw contractor strings are too high-cardinality for direct encoding. Derived features (repeat contractor flag, city) are more generalizable.

### 3.6 `Zip Code`

- 71 unique values — all start with `75xxx` (confirmed Dallas metro)
- **5,397 nulls (4.25%)** — manageable missing rate
- No non-Dallas zip codes detected

> **Decision:** Impute missing zip codes using modal zip per geographic proxy. Since we don't have `council_district`, we'll use `Mapsco` grid reference as the geographic grouping key for imputation. For records also missing Mapsco, impute with the overall mode.

### 3.7 `Land Use` (Occupancy Type equivalent)

- 189 unique categories
- Top 3: Single Family Dwelling (61.2%), Multi-Family Dwelling (8.2%), Office Building (5.0%)
- **No nulls** — complete field
- Strong value differentiation (commercial land uses have much higher permit values)

### 3.8 `Mapsco`

- Dallas-specific grid reference system (e.g., "39-N", "6-F,6-G,6-K,6-L")
- 3,066 unique values — high cardinality
- **890 nulls (0.7%)**
- Not referenced in the Execution Guide, but useful as a geographic proxy

> **Decision:** Use `Mapsco` for zip code imputation only. Drop before modeling — it's too granular and redundant with Zip Code.

### 3.9 `Work Description`

- 17,016 unique free-text descriptions
- Examples: "PERMANENT ELECTRICAL SERVICE", "NEW CONSTRUCTION", "INSTALL WATER HEATER"
- **902 nulls (0.71%)**
- Per the Execution Guide, this column should be **dropped before modeling** (too unstructured for tabular ML)

### 3.10 `Street Address`

- 76,065 unique values
- Per the Execution Guide, drop before modeling

---

## 4. Data Quality Summary

| Issue | Severity | Count / Rate | Action |
|---|---|---|---|
| Value = 0 | 🔴 High | 30,054 (23.7%) | Remove — not valid regression targets |
| Area = 0 | 🟡 Medium | 77,165 (60.8%) | Re-assess after Value-zero removal; impute by permit type |
| Zip Code missing | 🟢 Low | 5,397 (4.25%) | Impute using Mapsco-based mode |
| Work Description missing | 🟢 Low | 902 (0.71%) | Column will be dropped |
| Mapsco missing | 🟢 Low | 890 (0.70%) | Used only for imputation |
| Contractor missing | 🟢 Negligible | 3 (0.00%) | Drop these 3 rows |
| Street Address missing | 🟢 Negligible | 3 (0.00%) | Column will be dropped |
| `council_district` missing | 🔴 Structural | N/A — column absent | Drop from all pipeline steps |
| `Application Date` missing | 🔴 Structural | N/A — column absent | Use `Value` as target, not `Processing_Time` |
| Value extreme outliers | 🟡 Medium | $64.5M max | Cap at 99th percentile ($2M) |

---

## 5. Guide vs. Reality — Adaptation Decisions

| Guide Instruction | Actual Situation | Adapted Approach | Why |
|---|---|---|---|
| Pull from Socrata API | CSV already provided | Use CSV directly | Data already pulled |
| Target: `declared_value` | Column is `Value` (string w/ commas) | Parse commas → float | Data format difference |
| Impute zip by `council_district` | No `council_district` | Impute zip by `Mapsco` | Best geographic proxy available |
| One-hot encode `council_district` | Column absent | Skip | Cannot encode what doesn't exist |
| Compute `contractor_city` | Embedded in `Contractor` string | Parse with regex | City is after the address comma |
| `Processing_Time` target (Claude.md) | Only `Issued Date` exists | Target = `Value` → `log(Value)` | Per Execution Guide |
| Chronological split on `issue_date` | Sort by `Issued Date` | Identical logic, different column name | Direct adaptation |

---

## 6. Proposed Data Cleaning Plan (Phase 1, Step 3)

### Step 1: Load and Type-Cast
- Read CSV with all columns as strings
- Parse `Value` and `Area` → remove commas → cast to float
- Parse `Issued Date` → datetime (`%m/%d/%y` format)

### Step 2: Remove Unmodelable Rows
- Drop rows where `Value == 0` (expected: ~30,054 rows, 23.7%)
- Drop rows where `Value` is null (expected: 0)
- Drop rows where `Issued Date` is null (expected: 0)
- Drop rows where `Permit Type` is null (expected: 0)
- Drop rows where `Contractor` is null (expected: 3 rows)
- **Log: report exact counts dropped and percentage**

### Step 3: Parse Composite Fields
- Split `Permit Type` into `permit_category`, `permit_code`, `sub_category`, `work_type`
- Parse `Contractor` into `contractor_name`, `contractor_city`
- **Log: report parse success rate, list unparsed records**

### Step 4: Handle Missing Values
- `Zip Code` (4.25% null): impute with modal zip per `Mapsco` group; for records also missing Mapsco, use overall mode
- `Area` (re-check zero rate after Step 2): impute zeros with median Area grouped by `permit_category` + `sub_category` + `work_type`
- `Work Description` (0.71% null): column will be dropped — no imputation needed
- `Land Use` (0% null): no action needed

### Step 5: Outlier Handling
- Cap `Value` at the 99th percentile (~$2,000,000)
- Create `log_value` = log(Value) — this is the regression target
- **Log: report number of capped records**

### Step 6: Type Enforcement
- `Zip Code` → string (categorical, not numeric)
- `Value` and `Area` → float64
- `Issued Date` → datetime64
- All parsed permit components → string/categorical

### Step 7: Save & Report
- Save cleaned DataFrame as `permits_cleaned.csv`
- Report: final row count, columns, percentage of original data retained

---

## 7. Estimated Data Retention

| Stage | Rows | % of Original |
|-------|------|---------------|
| Raw data | 126,840 | 100% |
| After removing Value=0 | ~96,786 | ~76.3% |
| After removing null contractors | ~96,783 | ~76.3% |
| **Final clean dataset (est.)** | **~96,700+** | **~76%** |

> Retaining ~76% of the data is healthy for a production ML pipeline. The 24% removal is entirely justified — those records lack a meaningful prediction target.

---

## 8. Open Questions for Your Review

> [!IMPORTANT]
> **1. Council District:** The Execution Guide heavily references `council_district` but it's absent from the CSV. Should I:
> - **(A)** Drop it entirely from all pipeline steps (recommended for MVP speed), or
> - **(B)** Attempt to derive it from a Dallas open data lookup using Zip Code or Street Address?

> [!IMPORTANT]
> **2. Prediction Target Confirmation:** The Execution Guide targets `log(Value)` (declared construction value). The Claude.md MVP spec targets `Processing_Time` (application-to-issue days). Since we only have `Issued Date` (no Application Date), I will use `log(Value)` as the target. **Please confirm this is correct.**

> [!IMPORTANT]
> **3. Zero-Value Electrical Signs & Demolitions:** These are 100% zero-value permit types (combined ~7,400 rows). Removing them is the right call for regression, but do you want a **separate classification model** to predict permit type/category as a side deliverable?

---

## 9. Recommended Next Steps

1. **✅ Approve this analysis** and the cleaning plan above
2. **Phase 1 — Data Cleaning**: Execute Steps 1-7 above, producing `permits_cleaned.csv`
3. **Phase 1 — Feature Engineering**: Parse permit types, extract contractor features, create temporal features, encode categoricals
4. **Phase 2 — Modeling**: Baseline → Random Forest → XGBoost → SHAP → Clustering
5. **Phase 3 — Deployment**: Flask API → n8n Agent → Report
