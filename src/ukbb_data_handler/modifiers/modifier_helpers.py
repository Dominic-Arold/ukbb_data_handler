import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def get_field_columns(columns, field_id, instance= None):
    '''
    From a list of field column names, get all entries belonging to the given field.
    Parameters:
        columns : list[str]
            List of field column names
        field_id : str
            Field ID to search for, i.e. "<field_id>-"
        instance : int
            Optional. Search for specific instance of the field, i.e. "<field_id>-<instance>."
    '''
    search_str = f'{field_id}-' if instance==None else f'{field_id}-{instance}.'
    return [column for column in columns if column.startswith(search_str)]

def construct_columns_from_field(fields, instance, array=0):
    '''
    Construct the column string <field>-<instance>.<array> from given specifications for every given field.
    '''
    if isinstance(fields, int) or isinstance(fields, str):
        fields = [fields]
    elif isinstance(fields, list):
        pass
    else:
        log.error(f'Unexpected "fields" arg:\n{fields}')
    cols = [f'{f}-{instance}.{array}' for f in fields]
    return cols






