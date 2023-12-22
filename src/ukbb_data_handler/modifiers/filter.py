
def filter_instances(df, keep_instances : list, exceptions : list = []):
    '''
    Filter all fields to specified instances.
    Parameters:
        keep_instances : list
            List of instances to be kept.
        exceptions : list
            List of fields for which all instances will be kept.
    '''
    for _id in keep_instances:
        assert type(_id) == type(int())
    for _id in exceptions:
        assert type(_id) == type(int())

    original_fields = []
    df_columns = df.columns

    for c in df_columns: original_fields.append(int(c.split('-')[0]))
    for f_id in exceptions:
        assert f_id in original_fields
        # Please do not try to filter a field that doesn't exists in original df.
    field_ids = []
    for c in df_columns:
        f_id = int(c.split('-')[0])
        if f_id not in field_ids: field_ids.append(f_id)

    if len(exceptions) == 0:
        # This filters all of the fields
        columns_not_to_drop = []
        for f_id in field_ids:
            #_columns_to_drop = []
            for instance in keep_instances:
                for _column_name in df_columns:
                    if _column_name.startswith(f'{f_id}-{instance}'):
                        columns_not_to_drop.append(_column_name)

    else:
        # This does not drop any instances for the fields that are listed in exceptions
        columns_not_to_drop = []
        for f_id in field_ids:
            #_columns_to_drop = []
            for instance in keep_instances:
                for _column_name in df_columns:
                    if _column_name.startswith(f'{f_id}-{instance}'):
                        columns_not_to_drop.append(_column_name)
        
        for f_id in exceptions:
            for _column_name in df_columns:
                for instance in [0,1,2,3]:
                    if _column_name.startswith(f'{f_id}-{instance}'):
                        if _column_name not in columns_not_to_drop:
                            columns_not_to_drop.append(_column_name)              

    columns_to_drop = [c for c in df_columns if c not in columns_not_to_drop]
    df = df.drop(columns=columns_to_drop)
    return df

def drop_fields(df, fields: list=[], categories: list= []):
    '''
    '''
    from .modifier_helpers import get_field_columns

    for f in fields:
        drop_cols = get_field_columns(df.columns.values, f)
        df.drop(columns=drop_cols, inplace=True)
    if categories != []:
        print('Option categories not implemented.')
        pass
    return df
