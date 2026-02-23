# In[]:
!pip install seaborn
!mamba install pandas

# In[]:
# Dataset: Premier League Players

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis


# 1. Load Dataset

df = pd.read_csv("premier_league_players (1).csv")
print("First 5 rows of dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())


# 2. Data Cleaning

df = df.drop_duplicates()
df = df.dropna()

df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

df["Market Value"] = (
    df["Market Value"]
    .str.replace("€", "", regex=False)
    .str.replace("m", "e6", regex=False)
    .str.replace("k", "e3", regex=False)
)

df["Market Value"] = pd.to_numeric(df["Market Value"], errors="coerce")
df["Market Value (M€)"] = df["Market Value"] / 1e6

df = df.dropna()
print("\nData cleaned successfully. Total rows:", len(df))


# 3. RELATIONAL PLOT
#    Simple Scatter: Age vs Market Value

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    df["Age"],
    df["Market Value (M€)"],
    color="steelblue",
    s=40,
    alpha=0.6,
    edgecolors="white",
    linewidths=0.4
)

ax.set_title("Relational Plot — Age vs Player Market Value", fontsize=14, fontweight="bold")
ax.set_xlabel("Player Age (years)", fontsize=12)
ax.set_ylabel("Market Value (€ Millions)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# 4. CATEGORICAL PLOT
#    Bar: Average Market Value by Position

top_positions = df["Position"].value_counts().nlargest(8).index
df_top = df[df["Position"].isin(top_positions)]

pos_avg = (
    df_top.groupby("Position")["Market Value (M€)"]
    .mean()
    .reindex(top_positions)
)

fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.bar(
    pos_avg.index,
    pos_avg.values,
    color="steelblue",     # Single uniform color
    edgecolor="white",
    width=0.6
)

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.4,
        f"€{bar.get_height():.1f}M",
        ha="center", va="bottom", fontsize=9
    )

ax.set_title("Categorical Plot — Average Market Value by Position", fontsize=14, fontweight="bold")
ax.set_xlabel("Player Position", fontsize=12)
ax.set_ylabel("Average Market Value (€ Millions)", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# 5. STATISTICAL PLOT
#    Heatmap: Club Spending by Position

top_clubs     = df["Club"].value_counts().nlargest(10).index
top_positions = df["Position"].value_counts().nlargest(8).index

df_filtered = df[df["Club"].isin(top_clubs) & df["Position"].isin(top_positions)]

pivot = df_filtered.groupby(["Club", "Position"])["Market Value (M€)"].sum().unstack(fill_value=0)
pivot = pivot.reindex(columns=top_positions, fill_value=0)

fig, ax = plt.subplots(figsize=(14, 7))

im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Total Spending (€ Millions)", fontsize=11)

ax.set_xticks(range(len(pivot.columns)))
ax.set_yticks(range(len(pivot.index)))
ax.set_xticklabels(pivot.columns, fontsize=10, rotation=30, ha="right")
ax.set_yticklabels(pivot.index, fontsize=10)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val        = pivot.values[i, j]
        text_color = "white" if val > pivot.values.max() * 0.6 else "black"
        ax.text(
            j, i,
            f"€{val:.0f}M",
            ha="center",
            va="center",
            fontsize=9,
            color=text_color,
            fontweight="bold"
        )

for x in np.arange(-0.5, len(pivot.columns), 1):
    ax.axvline(x, color="white", linewidth=1.5)

for y in np.arange(-0.5, len(pivot.index), 1):
    ax.axhline(y, color="white", linewidth=1.5)

ax.set_title("Statistical Plot — Club Spending by Playing Position (€M)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Playing Position", fontsize=12)
ax.set_ylabel("Club", fontsize=12)

plt.tight_layout()
plt.show()


# 6. Four Statistical Moments

mean_val = df["Market Value (M€)"].mean()
var_val  = df["Market Value (M€)"].var()
skew_val = skew(df["Market Value (M€)"])
kurt_val = kurtosis(df["Market Value (M€)"])

print("\n4 Statistical Moments — Market Value")
print(f"1st Moment — Mean      : €{mean_val:.2f}M")
print(f"2nd Moment — Variance  : {var_val:.2f}")
print(f"3rd Moment — Skewness  : {skew_val:.4f}")
print(f"4th Moment — Kurtosis  : {kurt_val:.4f}")

# In[]:


