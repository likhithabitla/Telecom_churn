import pandas as pd

df = pd.read_csv('Data/processed/cleaned_data.csv')
print("Churn dtype:", df['Churn'].dtype)
print("Churn unique:", df['Churn'].unique()[:5])

df['Churn_label'] = df['Churn'].map({1: 'Churn', 0: 'No Churn'})
print("Churn_label unique:", df['Churn_label'].unique())
print("Any NaN in Churn_label:", df['Churn_label'].isna().any())

grp = df.groupby(['Contract','Churn_label']).size().reset_index(name='Count')
print("Groupby result:")
print(grp)
print("Churn_label values in grp:", grp['Churn_label'].unique())
