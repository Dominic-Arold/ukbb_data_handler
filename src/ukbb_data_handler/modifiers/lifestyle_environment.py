import pandas as pd
import numpy as np

from .modifier_helpers import construct_columns_from_field
from ..ukbb_data_handler import toggle_column_name

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def compute_alcohol_consumption(df, instance):
    '''
    Compute alcohol intake in standard ml per day from self-reported volumes of 
    consumption for various beverages. Uses Category 100051 from the Touchscreen
    interviews at assessment center visits.
    
    For conversion of standard volumes of beverages to standard units, see 
    https://www.nhs.uk/live-well/alcohol-advice/calculating-alcohol-units/
    https://alcoholchange.org.uk/alcohol-facts/interactive-tools/unit-calculator
    This is comparable to UKB conversion in other category
    https://biobank.ndph.ox.ac.uk/showcase/refer.cgi?tk=uBb2TQzYtdhzCS0HT8ILX1HpIWwUNaeS328456&id=437
    
    1558	Alcohol intake frequency.
    
    4407	Average monthly red wine intake
    4418	Average monthly champagne plus white wine intake
    4429	Average monthly beer plus cider intake
    4440	Average monthly spirits intake
    4451	Average monthly fortified wine intake
    4462	Average monthly intake of other alcoholic drinks
    
    1568	Average weekly red wine intake
    1578	Average weekly champagne plus white wine intake
    1588	Average weekly beer plus cider intake
    1598	Average weekly spirits intake
    1608	Average weekly fortified wine intake
    5364	Average weekly intake of other alcoholic drinks
    '''
    # conversion factors of UKB 'glasses' etc to standard units of alcohol. 
    # See https://www.nhs.uk/live-well/alcohol-advice/calculating-alcohol-units/
    ukb_wine_to_su = 1.5 # Small glass of red/white/rosé wine (125ml, ABV 12%) 1.5 units.
    ukb_beer_to_su = 2.5 # Pint of lower/higher-strength lager/beer/cider (ABV 3.6%) 2.0/3.0 units -> average
    # Single small shot of spirits (25ml, ABV 40%) 1 unit / Large (35ml) single measures of spirits are 1.4 units. 
    # And 1 bottle is 750 ml and UKBB says in one bottle there are 25 standard doses, 
    # then asks how many standard doses so 1 dose = 30ml -> average small and large measures
    ukb_spirit_to_su = 1.2 
    # average between large (1.3=75ml) and normal (0.9=50ml) portion of fortified wine
    # https://alcoholchange.org.uk/alcohol-facts/interactive-tools/unit-calculator
    ukb_fortified_to_su = 1.1
    # Alcopop (275ml, ABV 5.5%) 1.5 units
    ukb_alcopop_to_su = 1.5
    
    intake_col = f'alcohol_consumption-{instance}.0' # in ml/day
    intake_weekly_fields = [
        '1568',
        '1578',
        '1588',
        '1598',
        '1608',
        '5364',
    ]
    intake_monthly_fields = [
        '4407',
        '4418',
        '4429',
        '4440',
        '4451',
        '4462',
    ]
    intake_monthly_cols = construct_columns_from_field(intake_monthly_fields, instance)
    intake_weekly_cols = construct_columns_from_field(intake_weekly_fields, instance)
    intake_frequency_col = construct_columns_from_field(1558, instance)
    
    _df = df.loc[:, intake_frequency_col+intake_monthly_cols+intake_weekly_cols].copy()

    idx_monthly = _df.loc[_df[intake_monthly_cols].notna().any(axis=1), :].index
    idx_weekly = _df.loc[_df[intake_weekly_cols].notna().any(axis=1), :].index
    idx_never = _df.loc[(_df[intake_frequency_col]==6).values,:].index

    # participants were only asked either the weekly of monthly questions. Sanity check this here, since rely on it
    assert set(idx_monthly).intersection(idx_weekly) == set([])

    _df = _df.loc[_df[intake_frequency_col].isin([5,4,3,2,1]).values, :]
    _df.replace({-1:0.0, -3:0.0}, inplace=True) # "do not know/answer" for quantities of alcohol are counted as 0 here
    
    # All 6 beverage quantities should have a value or none of them (same for monthly consumption)
    # For weekly consumption this assertion fails since one beverage has NaN for most samples
    # "5364	Average weekly intake of other alcoholic drinks"
    # (and for one there are 5 NaN intake cols). Thus, set Nan to 0 and expect always all 6 values to be present
    _df[intake_weekly_cols] = _df[intake_weekly_cols].fillna(0.0)
    assert set(_df[intake_weekly_cols].isna().sum(axis=1).unique()) == set([0])
    weekly_consumption = \
        _df[construct_columns_from_field(1568, instance)].squeeze() * ukb_wine_to_su + \
        _df[construct_columns_from_field(1578, instance)].squeeze() * ukb_wine_to_su + \
        _df[construct_columns_from_field(1588, instance)].squeeze() * ukb_beer_to_su + \
        _df[construct_columns_from_field(1598, instance)].squeeze() * ukb_spirit_to_su + \
        _df[construct_columns_from_field(1608, instance)].squeeze() * ukb_fortified_to_su + \
        _df[construct_columns_from_field(5364, instance)].squeeze() * ukb_alcopop_to_su
    
    assert set(_df[intake_monthly_cols].isna().sum(axis=1).unique()) == set([6, 0])
    monthly_consumption = \
        _df[construct_columns_from_field(4407, instance)].squeeze() * ukb_wine_to_su + \
        _df[construct_columns_from_field(4418, instance)].squeeze() * ukb_wine_to_su + \
        _df[construct_columns_from_field(4429, instance)].squeeze() * ukb_beer_to_su + \
        _df[construct_columns_from_field(4440, instance)].squeeze() * ukb_spirit_to_su + \
        _df[construct_columns_from_field(4451, instance)].squeeze() * ukb_fortified_to_su + \
        _df[construct_columns_from_field(4462, instance)].squeeze() * ukb_alcopop_to_su
    
    # participant either answered montly or weekly intake questions
    df.loc[:,intake_col] = np.nan
    df.loc[idx_never, intake_col] = 0.0
    df.loc[idx_weekly, intake_col] = weekly_consumption[idx_weekly] / 7.0
    df.loc[idx_monthly, intake_col] = monthly_consumption[idx_monthly] / 4.3 / 7.0
    
    df[[intake_col]] *= 10.0
    return df

def IPAQ_MET_variables(df, instance):
    '''
    Calculates the continuous MET scores for IPAQ short form. See section 5.2 in UKB guideline:
    https://biobank.ndph.ox.ac.uk/ukb/ukb/docs/ipaq_analysis.pdf
    See www.ipaq.ki.se for more information about the score.
    By now (Jul 2023) these computed IPAQ fields are also available in the UKB showcase (Category 54).
    '''
    
    #duration_of_walks = 874
    #days_of_walks = 864
    #
    #duration_of_moderate_activity = 894
    #days_of_moderate_activity = 884
    #
    #duration_of_vigorous_activity = 914
    #days_of_vigorous_activity = 904
    fields = [874, 864, 894, 884, 914, 904]
    fields = construct_columns_from_field(fields, instance, array=0)
    df_ipaq = df.loc[df[fields].notna().any(axis=1), fields].copy()
    
    # logical child value imputation
    for i in range(3):
        f_duration = fields[2*i]
        f_days = fields[2*i+1]
        df_ipaq.loc[df_ipaq[f_days]==0.0, f_duration] = 0.0
    
    df_ipaq.loc[:, fields] = df_ipaq[fields].replace([-1,-2,-3], value=np.nan)
    
    for duration, frequency, level, weight in [(874, 864, 'walking', 3.3), 
                                               (894, 884, 'moderate', 4.0), 
                                               (914, 904, 'vigorous', 8.0)]:
        duration, frequency = construct_columns_from_field([duration, frequency], instance, array=0)
        df_ipaq[f'MET_{level}-{instance}.0'] = weight * df_ipaq[duration] * df_ipaq[frequency]
    
    df_ipaq[f'MET-{instance}.0'] = df_ipaq[f'MET_walking-{instance}.0'] + df_ipaq[f'MET_moderate-{instance}.0'] + df_ipaq[f'MET_vigorous-{instance}.0']
    # double check if 'other' (encoded < 0) responses were not computed into negative MET scores
    computed_fields = [
        f'MET_walking-{instance}.0', 
        f'MET_moderate-{instance}.0', 
        f'MET_vigorous-{instance}.0', 
        f'MET-{instance}.0'
    ]
    assert ((df_ipaq.loc[:, computed_fields] >= 0) | (df_ipaq.loc[:, computed_fields].isna())).all().all()

    # ---- activity groups -----
    col_group = f'IPAQ_group-{instance}.0'
    df_ipaq[col_group] = np.nan
    
    def at_least_sum(l, thr):
        '''
        Return bool whether cumulative sum in list l exceeds thr.
        In case of NaN in l, return NaN if sum of other entries does not already exceed thr
        since then the truth value is unknowable.
        '''
        at_least = np.nansum(l) >= thr
        if at_least==False and pd.isna(l).any():
            at_least = np.nan
        return at_least
    
    def AND_nan(*conditions):
        '''
        Consistent handling of logical AND when nan (i.e. not knowable) is involved.
        Just like normal AND, but return nan when at least one nan involved in conditions.
        '''
        conditions = pd.Series(conditions)
        if pd.isna(conditions).any():
            return np.nan
        return conditions.eq(True).all()
    
    def OR_nan(*conditions):
        '''
        Consistent handling of logical OR when nan (i.e. not knowable) is involved.
        Just like normal OR, but return nan when at least one nan involved in conditions
        and all others are False.
        '''
        conditions = pd.Series(conditions)
        if conditions.eq(True).any():
            return True
        elif conditions.eq(False).all():
            return False
        elif pd.isna(conditions).any() and (~conditions.eq(True).any()):
            return np.nan
        else:
            log.error(f'Unhandled input in {__name__}:\n{conditions}')
        
    def is_group_2(row):
        a = (row[f'Number of days/week of vigorous physical activity 10+ minutes-{instance}.0']>=3) & \
            (row[f'Duration of vigorous activity-{instance}.0']>=20)
        
        at_least_5 = at_least_sum([
                row[f'Number of days/week of moderate physical activity 10+ minutes-{instance}.0'],
                row[f'Number of days/week walked 10+ minutes-{instance}.0']
            ], 5)
        
        b1= AND_nan(at_least_5, row[f'Duration of moderate activity-{instance}.0']>=30)
        b2= AND_nan(at_least_5, row[f'Duration of walks-{instance}.0']>=30)
        
        at_least_5 = at_least_sum([
                row[f'Number of days/week of vigorous physical activity 10+ minutes-{instance}.0'],
                row[f'Number of days/week of moderate physical activity 10+ minutes-{instance}.0'],
                row[f'Number of days/week walked 10+ minutes-{instance}.0']
            ], 5)
        at_least_600_met = at_least_sum([
                row[f'MET_walking-{instance}.0'],
                row[f'MET_moderate-{instance}.0'],
                row[f'MET_vigorous-{instance}.0'],
            ], 600)
        c = AND_nan(at_least_5, at_least_600_met)
        return OR_nan(a,b1,b2,c)
    
    def is_group_3(row):
        at_least_1500_met = at_least_sum([
                row[f'MET_walking-{instance}.0'],
                row[f'MET_moderate-{instance}.0'],
                row[f'MET_vigorous-{instance}.0'],
            ], 1500)
        a = AND_nan(row[f'Number of days/week of vigorous physical activity 10+ minutes-{instance}.0']>=3, \
            at_least_1500_met)

        at_least_7 = at_least_sum([
                row[f'Number of days/week of vigorous physical activity 10+ minutes-{instance}.0'],
                row[f'Number of days/week of moderate physical activity 10+ minutes-{instance}.0'],
                row[f'Number of days/week walked 10+ minutes-{instance}.0']
            ], 7)
        at_least_3000_met = at_least_sum([
                row[f'MET_walking-{instance}.0'],
                row[f'MET_moderate-{instance}.0'],
                row[f'MET_vigorous-{instance}.0'],
            ], 3000)
        b = AND_nan(at_least_7, at_least_3000_met)
        return OR_nan(a,b)
    
    def set_group(row):
        grp2 = is_group_2(row)
        grp3 = is_group_3(row)
        l = [grp2,grp3]
        if pd.isna(l).any() and ~(pd.Series(l).eq(True).any()):
            return np.nan
        elif np.sum(l)==0:
            return 0
        elif grp3==True:
            return 2
        elif grp2==True:
            return 1
        else:
            print(f'Unexpected reuslts {l}')
    df_ipaq[col_group] = toggle_column_name(df_ipaq, to_id=False).apply(set_group, axis=1)
    
    idx = df_ipaq.index
    fields_all = fields + computed_fields + [col_group]
    df.loc[idx,fields_all] = df_ipaq
    return df