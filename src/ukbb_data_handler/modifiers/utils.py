import numpy as np
from .modifier_helpers import get_field_columns
from ..ukbb_data_handler import dask_apply_decorator

def collapse_array_index(df, fields, instances):
    '''
    For each given field (and per instance), put all array values as list in the first column (array=0).
    Example:
    Input dataframe
    53-1.0 | 53-1.1 | 53-1.2
    ------------------------
    A      | B      | C
    X      | Y      | Z

    Output dataframe
    53-1.0    |
    -----------
    [A,B,C]   |
    [X, Y, Z] |
    '''
    def _collapse_array(row):
        x = row.dropna().to_list()
        x = [i for i in x if i != 'nan'] # eg 41271-0.0 has 'nan' strings
        if len(x)==0:
            x=np.nan
        return x
    
    for f in fields:
        for i in instances:
            cols = get_field_columns(df.columns, f, instance=i)
            if len(cols)==0:
                continue
            vals = df[cols].apply(_collapse_array, axis=1)
            df.drop(columns=cols, inplace=True)
            df[f'{f}-{i}.0'] = vals
    return df

@dask_apply_decorator
def add_5(row, column_name, add=5):
    '''
    This is a dummy modifier to test the dask_apply_decorator. It gets a column and adds 5 every entry and sets up
    a new column for it. Also, keep in mind that the 'dask_apply_decorator' is defined in the main package file.
    So depending on how you import the package you may need to use it as @dh.dask_apply_decorator or similar.
    '''
    row[column_name] = row[column_name] + add
    return row