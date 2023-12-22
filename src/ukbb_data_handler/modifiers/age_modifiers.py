import numpy as np
import pandas as pd
from datetime import datetime
from dateutil import relativedelta

def age_at(df, field, instance):
    '''
    Calculates a new column containing each participant's age at a given reference date. 
    In most cases you might want to specify for the latter field 53 (Date of attending assessment centre) and instance 0-4.
    https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=53
    
    Resulting column will be named "Age_at_{field}-{instance}.0"
    
    Required field columns in df:
        34-0.0, 52-0.0 - Month of birth and Year of birth
    Parameters::
        field : int
            Field ID containing the reference dates.
        instance : int
            Concrete instance column of reference date for given field.
    '''

    col_date_ref = f'{field}-{instance}.0'
    col_age_at = f'Age_at_{col_date_ref}'
    
    def _compute_age_at(row):
        year_birth, month_birth = row["34-0.0"], row["52-0.0"]
        date_ref = row[col_date_ref]
        
        if (pd.notna(year_birth) and pd.notna(month_birth)) and pd.notna(date_ref):
            date_birth = datetime(year=int(year_birth), month=int(month_birth), day=15)
            date_ref = datetime.strptime(date_ref, "%Y-%m-%d")
            diff = relativedelta.relativedelta(date_ref, date_birth)
             # more than two digits unnecessary since we only know the exact day of birth up to half a month
            return round(diff.years + diff.months / 12.0 + diff.days / 365, 2)
        else:
            return np.nan

    df[col_age_at] = df.apply(_compute_age_at, axis=1)
    return df

def age_group(df, age_column, new_column_name = 'Age_Group'):
    '''
    Group age into sets of five year intervals. This modifier calculates a categorical column from a given age column.
    New categorical column has the entries 45to54, 55to64, 65to74, 75Plus, np.nan
    Parameters:
        age_column : str
            Column name containing age.
        new_column_name : str (default 'Age_Group')
            Optional. Name of created column of age groups.
    '''

    group_thrs = [0,45,50,55,60,65,70,75,80,1000]
    labels = [f'{group_thrs[i]}-{group_thrs[i+1]}' for i in range(len(group_thrs)-1)]
    df[new_column_name] = pd.cut(df[age_column].values, bins=group_thrs, labels=labels)
    return df
