import pandas as pd
import numpy as np

# dog owner csv

df = pd.read_csv('DAP_2021_HLES_dog_owner_v1.0.csv')

print(df.shape)

index_df = df.set_index('dog_id')
exp_df = index_df.loc[:,
         ["dd_age_years", 'dd_sex', 'hs_health_conditions_liver', 'dd_weight_lbs', 'dd_breed_pure_or_mixed',
          'dd_breed_pure', 'dd_spayed_or_neutered', 'df_diet_consistency', 'df_feedings_per_day',
          'df_daily_supplements_omega3', 'df_primary_diet_component', 'hs_general_health']]
print(exp_df.shape)

"""#unwanted variables exclusion

Reason for exclusion:
1. age selected between 1 to 18. As below 1 year, puppies do not have fixed feeding frequency compared to adults and 
   above 18 years older dogs there are outliers
2. only neutered dogs are considered as almost 95% were spayed.
3. Also exlcuded dogs which didnt have a consistent feeding frequency at all as we want to analyze the effect of 
   frequency
"""

clean = (exp_df["dd_age_years"] >= 1) & (exp_df["dd_age_years"] < 18) & (exp_df['dd_spayed_or_neutered'] == True) & (
        exp_df['df_diet_consistency'] != 3)
data = exp_df[clean]
print(data.head())

print(data.shape)

"""# Excluding the Sample if the Value count of any breed is less than 10

"""

data.replace(np.nan, 0, inplace=True)

value_counts = data['dd_breed_pure'].value_counts()

# Loop through the unique values in the 'FloatColumn' and delete rows where count is 0
for value in value_counts.index:
    if value_counts[value] < 10:
        data = data[data['dd_breed_pure'] != value]

"""#health conditions csv"""

df1 = pd.read_csv('DAP_2021_HLES_health_conditions_v1.0.csv')
df2 = df1.set_index('dog_id')
hc = df2.loc[:, ['hs_condition_type', 'hs_condition', 'hs_condition_is_congenital']]

print(hc.shape)

"""#merging health condiiton variable from health condition csv file"""

final = pd.merge(data, hc, on='dog_id', how='inner', copy=True)
print(final.head())
print(final.shape)
