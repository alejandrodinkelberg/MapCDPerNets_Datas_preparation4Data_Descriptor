import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

# -----------------------------
# PATHS
# -----------------------------
input_path = r"CSVFILE"

output_dir = r"DIRECTORY_FOR_OUTPUT"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(input_path)

if "AutoAccept" in df.columns:
    df = df[df["AutoAccept"] == True]

# -----------------------------
# AGE GROUPS
# -----------------------------
bins = [18, 25, 35, 45, 55, 65, 120]
labels = ["18–24", "25–34", "35–44", "45–54", "55–64", "65+"]

df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

# -----------------------------
# TABLE TOTAL (WITH FULL GENDER BREAKDOWN)
# -----------------------------
n_total = len(df)

female_pct = (df["gender"] == 2).mean() * 100
male_pct = (df["gender"] == 1).mean() * 100
other_pct = 100 - female_pct - male_pct  # includes NA / other

table_total = pd.DataFrame({
    "N_total": [n_total],
    "mean_age": [round(df["age"].mean(), 1)],
    "sd_age": [round(df["age"].std(), 1)],
    "min_age": [df["age"].min()],
    "max_age": [df["age"].max()],
    "n_female": [(df["gender"] == 2).sum()],
    "n_male": [(df["gender"] == 1).sum()],
    "female_pct": [round(female_pct, 1)],
    "male_pct": [round(male_pct, 1)],
    "other_pct": [round(other_pct, 1)]
})

table_total.to_csv(os.path.join(output_dir, "table_total.csv"), index=False)

# -----------------------------
# COUNTRY TABLE (WITH FULL GENDER BREAKDOWN)
# -----------------------------
country_table = (
    df.groupby("country")
    .apply(lambda g: pd.Series({
        "N": len(g),
        "mean_age": round(g["age"].mean(), 1),
        "sd_age": round(g["age"].std(), 1),
        "min_age": g["age"].min(),
        "max_age": g["age"].max(),
        "n_female": (g["gender"] == 2).sum(),
        "n_male": (g["gender"] == 1).sum(),
        "female_pct": round((g["gender"] == 2).mean() * 100, 1),
        "male_pct": round((g["gender"] == 1).mean() * 100, 1),
        "other_pct": round(
            100
            - (g["gender"] == 2).mean() * 100
            - (g["gender"] == 1).mean() * 100,
            1
        )
    }))
    .reset_index()
)

country_table = country_table.sort_values("country")
country_table.to_csv(os.path.join(output_dir, "table_country.csv"), index=False)


# -----------------------------
# AGE GROUP TABLE (NEW)
# -----------------------------
age_table = df["age_group"].value_counts().sort_index().to_frame("N")
age_table["pct"] = (age_table["N"] / age_table["N"].sum() * 100).round(1)
age_table.to_csv(os.path.join(output_dir, "age_groups_table.csv"))

# -----------------------------
# AGE GROUP FIGURE (PERCENT)
# -----------------------------
plt.figure()

age_pct = age_table["pct"]

sns.barplot(x=age_pct.index, y=age_pct.values)

plt.xlabel("Age group")
plt.ylabel("Respondents (%)")
plt.ylim(-0.5, 30)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "age_groups.png"), dpi=300)
plt.close()

# -----------------------------
# ALTER DATA (LONG)
# -----------------------------
alter_list = []

for i in range(1, 31):
    tmp = df[[
        f"alter{i}_gender",
        f"alter{i}_age",
        f"alter{i}_closeness",
        f"alter{i}_relationship_type"
    ]].copy()

    tmp.columns = ["gender", "age", "closeness", "relationship"]
    alter_list.append(tmp)

df_alters = pd.concat(alter_list, ignore_index=True)

# -----------------------------
# RELATIONSHIP LABELS
# -----------------------------
rel_map = {
    1: "Family",
    2: "Neighbors",
    3: "Work",
    4: "Higher education",
    9: "School",
    10: "Through others",
    11: "Hobbies",
    12: "Other"
}

df_alters["relationship_label"] = df_alters["relationship"].map(rel_map)

# -----------------------------
# ALTER SUMMARY (WITH COUNT + FULL GENDER BREAKDOWN)
# -----------------------------
n_total = len(df_alters)

female_pct = (df_alters["gender"] == 2).mean() * 100
male_pct = (df_alters["gender"] == 1).mean() * 100

# everything else (including NA) → OTHER
other_pct = 100 - female_pct - male_pct

alter_summary = pd.DataFrame({
    "N_alters": [n_total],
    "mean_age": [round(df_alters["age"].mean(), 2)],
    "sd_age": [round(df_alters["age"].std(), 2)],
    "mean_closeness": [round(df_alters["closeness"].mean(), 2)],
    "female_pct": [round(female_pct, 1)],
    "male_pct": [round(male_pct, 1)],
    "other_pct": [round(other_pct, 1)]
})

alter_summary.to_csv(os.path.join(output_dir, "alter_summary.csv"), index=False)

# -----------------------------
# ALTER RELATIONSHIP TABLE (NEW)
# -----------------------------
rel_table = df_alters["relationship_label"].value_counts().to_frame("N")
rel_table["pct"] = (rel_table["N"] / rel_table["N"].sum() * 100).round(1)
rel_table.to_csv(os.path.join(output_dir, "alter_relationships_table.csv"))

# -----------------------------
# ALTER CLOSENESS FIGURE (PERCENT, DISCRETE)
# -----------------------------
plt.figure()

closeness_pct = (
    df_alters["closeness"]
    .value_counts(normalize=True)
    .sort_index() * 100
)

sns.barplot(x=closeness_pct.index, y=closeness_pct.values)

plt.xlabel("Emotional closeness (1–5)")
plt.ylabel("Alters (%)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "alter_closeness.png"), dpi=300)
plt.close()

# -----------------------------
# ALTER RELATIONSHIP FIGURE (PERCENT + COLOR)
# -----------------------------
plt.figure()

rel_pct = rel_table["pct"]

sns.barplot(
    x=rel_pct.values,
    y=rel_pct.index,
    palette="deep"
)

plt.xlabel("Alters (%)")
plt.ylabel("Relationship type")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "alter_relationships.png"), dpi=300)
plt.close()
