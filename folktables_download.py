import pandas as pd
from folktables import ACSDataSource, ACSIncome

state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']

year_list = [2018]

total = 0
total_df = pd.DataFrame(columns = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP',
                    'RELP', 'WKHP', 'SEX', 'RAC1P', 'TARGET'])


for state in state_list:
    for year in year_list:
        data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
        data = data_source.get_data(states=[state], download=True)

        features, labels, _ = ACSIncome.df_to_numpy(data)
        df = pd.DataFrame(features, columns=total_df.columns[:-1])
        df["TARGET"] = labels

        total_df = pd.concat((total_df, df))

        total += features.shape[0]

total_df.to_csv('data/data.csv', index=False)
