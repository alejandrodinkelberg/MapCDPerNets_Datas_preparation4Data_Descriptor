import pandas as pd
from pathlib import Path
import numpy as np

# -----------------------------
# FILE PATHS
# -----------------------------
file_main = Path(r"CSVFILE")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(file_main, dtype=str)

# If ID exists but has missing values, fill from first column
df["ID"] = df["ID"].fillna(df.iloc[:,0])
df["ID"] = df["ID"].astype(str).str.strip()


# -----------------------------
# KEEP ONLY SELECTED COLUMNS
# -----------------------------
keep_cols = [
"ID","Country","StartDate","Duration (in seconds)","Q245","Q246","Q247",
"Respuesta Consent","DataDon_consent","Q2","Q3","Q4_clean","Etnolinguistico_clean",
"Q225","Q227","Q227_5_TEXT_clean","Q230","Q230_5_TEXT_clean","A","B",
"C_1","C_2","C_3","C_4","C_5","C_6","AutoAccept"
]


def rename_all_variables(df):
    """
    Renames ego-level, alter-level, and alter–alter tie variables
    into consistent, informative snake_case names.
    Returns updated df.
    """

    df = df.copy()
    keep_cols = []

    # -------------------------
    # 1. EGO-LEVEL VARIABLES
    # -------------------------
    rename_dict = {
        "Country": "country",
        "StartDate": "survey_start_time",
        "Duration (in seconds)": "survey_duration_sec",

        "Q245": "social_media_platform",
        "Q246": "social_media_active_2yrs",
        "Q247": "willing_to_donate_data",

        "Respuesta Consent": "consent_general",
        "DataDon_consent": "consent_data_donation",

        "Q2": "gender",
        "Q3": "age",
        "Q4_clean": "residence_clean",
        "Etnolinguistico_clean": "ethnolinguistic_group_clean",

        "Q225": "household_size",
        "Q227": "household_composition",
        "Q227_5_TEXT_clean": "household_composition_other_clean",

        "Q230": "marital_status",
        "Q230_5_TEXT_clean": "marital_status_other_clean",

        "A": "A_decision_strategy",
        "B": "B_child_values",

        "C_1": "C_1_norm_prevalence",
        "C_2": "C_2_norm_clarity",
        "C_3": "C_3_norm_consensus",
        "C_4": "C_4_norm_behavioral_freedom",
        "C_5": "C_5_norm_sanctions",
        "C_6": "C_6_norm_compliance",
    }

    df = df.rename(columns=rename_dict)

    # keep ego vars
    keep_cols.extend([v for v in rename_dict.values() if v in df.columns])

    # -------------------------
    # 2. ALTER ATTRIBUTES
    # -------------------------
    attr_map = {
        "Género": "gender",
        "Edad": "age",
        "Residencia": "residence",
        "Cercanía": "closeness",
        "Tipo de relación": "relationship_type"
    }

    for i in range(1, 31):
        for orig, new in attr_map.items():
            old_col = f"{i}_{orig}"
            new_col = f"alter{i}_{new}"

            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
                keep_cols.append(new_col)

    # -------------------------
    # 3. ALTER–ALTER TIES
    # -------------------------

    # --- Q188 block (alter1 vs others, reverse order) ---
    for j in range(1, 30):
        old_col = f"Q188_{j}"

        # 🔥 reverse mapping
        target = 30 - (j - 1)

        a, b = sorted([1, target])
        new_col = f"alter{a}_alter{b}_tie"

        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
            keep_cols.append(new_col)

    # --- Remaining triangular blocks ---
    n_items = 29
    for i in range(1, n_items):
        focal = i + 1

        for j in range(1, n_items + 1 - i):
            old_col = f"Q{197 + i - 1}_{j}"

            # 🔥 reverse mapping
            target = 30 - (j - 1)

            a, b = sorted([focal, target])
            new_col = f"alter{a}_alter{b}_tie"

            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
                keep_cols.append(new_col)

    # -------------------------
    # OPTIONAL: remove duplicates
    # -------------------------
    keep_cols = list(dict.fromkeys(keep_cols))

    return df

def filter_relevant_variables(df):
    df = df.copy()

    # -------------------------
    # DROP EXACT COLUMNS
    # -------------------------
    drop_exact = [
        # IDs / metadata
        "ResponseId", "ExternalReference", "ContactID",
        "StartDate", "EndDate", "RecordedDate", "Progress",
        "Duration (in seconds)", "Source File", "Column Info",
        "Latitude", "Longitude",

        # consent / process
        "consent_general", "consent_data_donation",
        "social_media_platform",
        "social_media_active_2yrs",

        # duplicates / raw text
        "Q4", "Etnolinguistico", "Q227_5_TEXT", "Q230_5_TEXT",

        # alter names (not needed)
        *[f"ALF30_{i}" for i in range(1, 31)],

        # admin / QA
        "Empty Fields Count", "name_validation",
        "cercania_90_same", "tipo_90_same", "egdes_90_same",
        "edge_count_not_complete", "Any_Suspicious",
        "Q_UnansweredPercentage", "Q_UnansweredQuestions",
        "Q_AmbiguousTextPresent", "Q_AmbiguousTextQuestions",
        "Q_StraightliningCount", "Q_StraightliningPercentage",
        "Q_StraightliningQuestions",
        "is_junk", "is_none",

        # duplicates / IDs
        "Email", "Email.1", "user_id", "user_id_UNUSED",
        "uid", "sid_id", "PROLIFIC_PID", "Prolific ID_1",

        # misc
        "Source", "informed agreement", "data_donation_accepted",
        "SC0", "Q81", "smod_code", "Commitment to survey", "urban_rural",
        
        # Timestamp components (if not needed)
        "survey_start_time",

        # AutoAccept
        "AutoAccept"


    ]

    df = df.drop(columns=[c for c in drop_exact if c in df.columns], errors="ignore")

    return df

def harmonize_willingness(df):

    print("Before:")
    print(df["willing_to_donate_data"].value_counts(dropna=False))

    df = df.copy()

    # -------------------------
    # Clean columns FIRST
    # -------------------------
    df["country"] = df["country"].astype(str).str.strip()

    df["consent_data_donation"] = pd.to_numeric(
        df["consent_data_donation"], errors="coerce"
    )

    df["consent_general"] = pd.to_numeric(
        df["consent_general"], errors="coerce"
    )

    df["data_donation_accepted"] = (
        df["data_donation_accepted"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # -------------------------
    # USA
    # -------------------------
    mask_us = df["country"].isin(["US", "USA","JP", "Japan", "JA"])

    df.loc[
        mask_us & (
            (df["consent_data_donation"] == 1) |
            (df["data_donation_accepted"].isin(["true", "1"]))
        ),
        "willing_to_donate_data"
    ] = 1

    # -------------------------
    # Spain (EMAIL proxy)
    # -------------------------
    mask_es = df["country"].isin(["SP", "Spain", "ES"])
    # reset first
    df.loc[mask_es, "willing_to_donate_data"] = np.nan
    if "Email" in df.columns:
        has_email = df["Email"].notna() & (df["Email"].astype(str).str.strip() != "")
        df.loc[
            mask_es & has_email,
            "willing_to_donate_data"
        ] = 1

    # -------------------------
    # Romania
    # -------------------------
    mask_ro = df["country"].isin(["RO", "Romania"])

    # Ensure datetime
    df["RecordedDate"] = pd.to_datetime(df["RecordedDate"], errors="coerce", dayfirst=True)

    cutoff_date = pd.Timestamp("2025-07-22")

    # Condition: valid Romania willingness
    mask_valid_ro = (
        mask_ro &
        (df["consent_general"] == 1) &
        (
            (df["RecordedDate"] <= cutoff_date) |  # before cutoff → ok
            (df["consent_data_donation"].notna())  # after cutoff → must have consent_data_donation
        )
    )
    df.loc[mask_valid_ro, "willing_to_donate_data"] = 1


    # -------------------------
    # FINAL CLEAN: enforce binary
    # -------------------------
    df["willing_to_donate_data"] = pd.to_numeric(
        df["willing_to_donate_data"], errors="coerce"
    )

    df.loc[~df["willing_to_donate_data"].isin([1]), "willing_to_donate_data"] = np.nan

    df["willing_to_donate_data"] = df["willing_to_donate_data"].astype("Int64")

    print("\nAfter:")
    print(df["willing_to_donate_data"].value_counts(dropna=False))

    return df

def drop_underage(df, age_col="age"):
    df = df.copy()

    # ensure numeric age
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")

    # mask under 18
    mask_under18 = df[age_col] < 18

    # print rows that will be removed
    print("\n--- Participants younger than 18 (to be removed) ---")
    if mask_under18.sum() == 0:
        print("None found.")
    else:
        print(df.loc[mask_under18, ["ID", "country", age_col]].sort_values(by=age_col))

    print(f"\nTotal removed: {mask_under18.sum()}")

    # drop them
    df = df.loc[~mask_under18].copy()

    return df
# ALF30 columns
keep_cols += [f"ALF30_{i}" for i in range(1,31)]

# alter attributes
attrs = ["Género","Edad","Residencia","Cercanía","Tipo de relación"]
for i in range(1,31):
    for a in attrs:
        keep_cols.append(f"{i}_{a}")

# alter–alter network matrix
for q in range(188,225):
    for k in range(1,31):
        keep_cols.append(f"Q{q}_{k}")

# keep only columns that actually exist
keep_cols = [c for c in keep_cols if c in df.columns]

df_main = df[keep_cols]

# --------------------------------
# ANONYMIZE ALTER NAMES
# --------------------------------
for i in range(1, 31):
    col = f"ALF30_{i}"
    if col in df.columns:
        df[col] = f"ALTER{i}"

# --------------------------------
# FIX AutoAccept columns
# --------------------------------
df['AutoAccept'] = df['AutoAccept'].astype(str)
df = df[df['AutoAccept']=="True"]


# --------------------------------
# CREATE COUNTRY-SPECIFIC IDS
# --------------------------------
def create_country_specific_ids(df):
    base_start = 10001

    country_offsets = {
        "MX": 0,
        "FR": 1000,
        "SA": 2000,
        "MO": 3000,
        "SW": 4000,
        "RO": 5000,
        "SP": 6000,
        "JP": 7000,
        "GR": 8000,
        "US": 9000
    }

    df = df.reset_index(drop=True)

    new_ids = []
    country_counts = {}

    for _, row in df.iterrows():
        c = row["Country"]

        if c not in country_counts:
            country_counts[c] = 0

        num = base_start + country_offsets.get(c, 0) + country_counts[c]
        new_ids.append(f"{c}_{num}")

        country_counts[c] += 1

    df["ID"] = new_ids

    return df


# -----------------------------
# FINAL COLUMNS
# -----------------------------
df = create_country_specific_ids(df)
df = rename_all_variables(df)
df = drop_underage(df)   # ROx2, JPx1, SAx1

df = harmonize_willingness(df)
df = filter_relevant_variables(df)

# ------------------------------
# Define column order
# ------------------------------
# -----------------------------
# FINAL COLUMN ORDER
# -----------------------------
core_order = [
    "ID",
    "country",
    "survey_duration_sec",
    "willing_to_donate_data",
    "gender",
    "age",
    "residence_clean",
    "ethnolinguistic_group_clean",
    "Etnolinguistico_simple",
    "household_size",
    "household_composition",
    "household_composition_other_clean",
    "marital_status",
    "marital_status_other_clean",
    "A_decision_strategy",
    "B_child_values",
    "C_1_norm_prevalence",
    "C_2_norm_clarity",
    "C_3_norm_consensus",
    "C_4_norm_behavioral_freedom",
    "C_5_norm_sanctions",
    "C_6_norm_compliance"
]

# keep only existing
core_order = [c for c in core_order if c in df.columns]

remaining = [c for c in df.columns if c not in core_order]

df = df[core_order + remaining]

# -----------------------------
# SAVE
# -----------------------------
out_path = Path(r"CSVFILE")

df.to_csv(out_path, index=False)

print("Merged dataset saved to:")
print(out_path)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])