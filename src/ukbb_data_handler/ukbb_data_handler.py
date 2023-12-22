import os
import pandas as pd
import dask.dataframe as dd
from datetime import timedelta
from timeit import default_timer as timer
from multiprocessing import Pool
import functools
from functools import partial

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def _get_cache_fname(basket_id, field_id):
    '''
    Each field from a particular data basket is cached for quicker access in the future.
    This function provides a consistent file name pattern.
    '''
    return f'BASKETID_{basket_id}-FIELDID_{field_id}.feather'

def _get_showcase_mapping(category_ids=[], field_ids=[]):
    """
    Reads SHOWCASE_MAPPING_FILE to parse rows of requested fields and categories.
    Parameters:
        category_ids:
            requested category ids as a list
        field_ids:
            requested extra field ids as a list
    """
    showcase_mapping_tsv_path = os.environ.get('SHOWCASE_MAPPING_FILE')
    f_tsv = pd.read_csv(showcase_mapping_tsv_path, sep='\t')
    f_tsv = f_tsv.loc[f_tsv['Category'].isin(category_ids) | f_tsv['FieldID'].isin(field_ids)]
    return f_tsv

def cache_fields(df, field_list, basket_id, dir_path):
    """
    Save field columns from a dataframe as feather file into the UKBB_CACHE_DIR directory.

    Parameters:
        df : pandas.DataFrame
            Dataframe containing columns in field_list
        field_list : list
            List of fields to be cached.
        basket_id : str
            ID of the basket which is used (to construct file name)
        dir_path : str
            Absolute path of the cache folder.
    """
    cn = df.columns
    for f_id in field_list:
        f_instances = []
        for c in cn: 
            if str(f_id) == c.split('-')[0]: f_instances.append(c)
        field_df = df[f_instances]
        field_df_name = _get_cache_fname(basket_id, f_id)
        fpath = os.path.join(dir_path, field_df_name)
        try:
            field_df.reset_index().to_feather(fpath) # TODO: feather has problems loading when there are list-like as values
        except:
            for c in field_df.columns:
                field_df[c] = field_df[c].astype(str)
            field_df.reset_index().to_feather(fpath) # TODO: feather has problems loading when there are list-like as values

def get_fields_from_cache(field_list, basket_id, dir_path):
    """
    Parse cached feather files produced with cache_fields and return as dataframe.
    
    Parameters:
        field_list : list
            List of fields to be cached.
        basket_id : str
            ID of the basket which is used (to construct file name)
        dir_path : str
            Absolute path of the cache folder.
    """
    df_list = []
    for f_id in field_list:
        field_df_name = _get_cache_fname(basket_id, f_id)
        fpath = os.path.join(dir_path, field_df_name)
        df_list.append(pd.read_feather(fpath).set_index("eid"))
    df = pd.concat(df_list, axis=1)
    return df


def toggle_column_name(df, to_id=True):
    """
    Toggle between field IDs and corresponding clear names in column names of a dataframe.
    Example: '21001-0.0' <-> 'Body mass index (BMI)-0.0'.

    Parameters:
        df : pandas.DataFrame
            Dataframe containing field columns.
        to_id : bool
            If True (default) turns field names to field ids, if False the other way around.
    """
    
    showcase_mapping_tsv_path = os.environ.get('SHOWCASE_MAPPING_FILE')
    df_mapping_parsed_fields = pd.read_csv(showcase_mapping_tsv_path, sep='\t')
    
    rename_dict = {}
    f_tsv_mapping = {}
    columns = df.columns
    field_id_name = df_mapping_parsed_fields[['FieldID', 'Field']].values
    if to_id:
        for _id, _name in field_id_name: f_tsv_mapping[str(_name)] = str(_id)
    else:
        for _id, _name in field_id_name: f_tsv_mapping[str(_id)] = str(_name)
        
    for c in columns:
        try:
            name_id, instance = c.rsplit('-',1)
            rename_dict[c] = f_tsv_mapping[name_id] + '-' + instance
        except:
            continue
    df = df.rename(columns=rename_dict)
    return df

def get_withdrawn_eids():
    '''
    According to UKBB rules, participants who have wished to withdraw from the study must be removed from any future
    analysis. The IDs (eid) of these participants are communicated through periodic emails containing CSV files.
    Whenever receiving such an email, just put the attached CSV file in the directory specified by the
    PARTICIPANT_WITHDRAWALS_DIR environment variable. If you set PARTICIPANT_WITHDRAWALS_DIR to None, no samples will
    be removed (not recommended).
    Returns a set of found eid values.
    '''
    with_drawn_path = os.environ.get('PARTICIPANT_WITHDRAWALS_DIR')
    withdrawn_eids = set()
    if with_drawn_path is not None:
        for file_name in os.listdir(with_drawn_path):
            withdrawn_eids |= set(pd.read_csv(os.path.join(with_drawn_path, file_name), names = ['eid'])['eid'])
    return withdrawn_eids


def download_encodings(cache):
    """
    Downloads necessary files for toggling between encoded/decoded  field values.
    This function needs to be run at least once in order for column_raw_to_labeled and toggle_encoding to work.

    Parameters:
        cache : str
            Directory path to store downloaded files.
    """
    command_list = [
        f'''wget -q -nd -O {cache}/category.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=3"''',
        f'''wget -q -nd -O {cache}/catbrowse.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=13"''',
        f'''wget -q -nd -O {cache}/record_column.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=18"''',
        f'''wget -q -nd -O {cache}/field.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=1"''',
        f'''wget -q -nd -O {cache}/recommended.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=14"''',
        f'''wget -q -nd -O {cache}/fieldsum.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=16"''',
        f'''wget -q -nd -O {cache}/encoding.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=2"''',
        f'''wget -q -nd -O {cache}/snps.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=15"''',
        f'''wget -q -nd -O {cache}/instances.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=9"''',
        f'''wget -q -nd -O {cache}/publication.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=19"''',
        f'''wget -q -nd -O {cache}/record_table.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=17"''',
        f'''wget -q -nd -O {cache}/returns.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=4"''',
        f'''wget -q -nd -O {cache}/schema.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=999"''',
        f'''wget -q -nd -O {cache}/ehierint.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=11"''',
        f'''wget -q -nd -O {cache}/ehierstring.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=12"''',
        f'''wget -q -nd -O {cache}/insvalue.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=10"''',
        f'''wget -q -nd -O {cache}/esimpdate.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=8"''',
        f'''wget -q -nd -O {cache}/esimpint.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=5"''',
        f'''wget -q -nd -O {cache}/esimpreal.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=7"''',
        f'''wget -q -nd -O {cache}/esimpstring.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=6"''',
        f'''wget -q -nd -O {cache}/esimptime.txt "biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=20"''',
    ]
    for command in command_list:
        os.system(command)


def column_raw_to_labeled(df, field_column, is_name=True, dir_path=None, cached=False, convert_back=False):
    '''
    Toggles the entries of given column between human readable and default encoded values.
    
    Parameters:
    df : pandas.DataFrame
        Dataframe containing the specified field column
    field_column : str
        A column name in df. Can be either clear name or field id, as handled with toggle_column_name().
    is_name : bool (default True)
        if field_column is clear name set this to True, else False
    dir_path : str (default None)
        Optional path to directory containing coding files. If None uses environment variable UKBB_CACHE_DIR.
    cached : bool (default False)
        If False downloads the necessary encoding files in order to convert entries.
    convert_back : bool (default False)
        If True converts from human readable to encoded values.
    '''
    if dir_path is None:
        dir_path = os.environ.get('UKBB_CACHE_DIR')
        
    if not cached:
        encoding_dicts = [
            'category', 'catbrowse', 'record_column', 'field', 'recommended', 
            'fieldsum', 'encoding', 'snps', 'instances', 'publication', 'record_table',
            'returns', 'schema', 'ehierint', 'ehierstring', 'insvalue', 'esimpdate', 'esimpint',
            'esimpreal', 'esimpstring', 'esimptime',        
        ]
        cached_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        download = False
        for file in encoding_dicts:
            if f'{file}.txt' not in cached_files:
                download = True
                break
        if download:
            download_encodings(dir_path)

    field = pd.read_csv(f'{dir_path}/field.txt', sep='\t', encoding='latin-1')
    #encoding = pd.read_csv(f'{dir_path}/encoding.txt', sep='\t', encoding='latin-1')
    ehierint = pd.read_csv(f'{dir_path}/ehierint.txt', sep='\t', encoding='latin-1')
    ehierstring = pd.read_csv(f'{dir_path}/ehierstring.txt', sep='\t', encoding='latin-1')
    #insvalue = pd.read_csv(f'{dir_path}/insvalue.txt', sep='\t', encoding='latin-1')
    esimpdate = pd.read_csv(f'{dir_path}/esimpdate.txt', sep='\t', encoding='latin-1')
    esimpint = pd.read_csv(f'{dir_path}/esimpint.txt', sep='\t', encoding='latin-1')
    esimpreal = pd.read_csv(f'{dir_path}/esimpreal.txt', sep='\t', encoding='latin-1')
    esimpstring = pd.read_csv(f'{dir_path}/esimpstring.txt', sep='\t', encoding='latin-1')
    esimptime = pd.read_csv(f'{dir_path}/esimptime.txt', sep='\t', encoding='latin-1')

    if is_name:
        condition = 'title'
        condition_value = str(field_column.split('-')[0])
    else:
        condition = 'field_id'
        condition_value = int(field_column.split('-')[0])
    possible_fields = field[condition].values
    if condition_value in possible_fields:
        log.info(f'Converting values for field {field_column}.')
        encoding_id = field.loc[field[condition] == condition_value, 'encoding_id'].values[0]
    else:
        log.warning(f'The field {field_column} is not in the fields listed in the field.txt file')
        return 0

    list_of_series = []
    for ddict in [ehierint, ehierstring, esimpdate, esimpreal, esimpstring, esimptime, esimpint]:
        if encoding_id in ddict['encoding_id'].unique():
            _series = df[field_column].copy()
            coding = dict(zip(ddict.loc[ddict['encoding_id'] == encoding_id, 'value'].values, ddict.loc[ddict['encoding_id'] == encoding_id, 'meaning'].values))
            if convert_back:
                _series = _series.replace({v: k for k, v in coding.items()})
            else:
                _series = _series.replace(coding)
            list_of_series.append(_series)

    if len(list_of_series) > 1:
        log.error(f'The field {field_column} for "column_raw_to_labeled" is in more than one dictionary, returning the first one whichever that is!!')
        return list_of_series[0]
    elif len(list_of_series) <= 0:
        log.error(f'the field {field_column} for "column_raw_to_labeled" is in none of the dictionaries!!!')
        return 0
    else:
        return list_of_series[0]

def toggle_encoding(df, is_name=True, dir_path=None, convert_back=False):
    """
    Toggles all values in df between human readable decoded and encoded form.
    
    Parameters:
    df : pandas.DataFrame
        Output of fetch_ukb_main function.
    is_name : bool (default True)
        Column names are used to find the right encoding dict. If column name is in clear name set this to True,
        otherwise to False for field id column names (see toggle_column_name function).
    dir_path : str (default None)
        Path of cache directory, if None uses environment variable UKBB_CACHE_DIR.
    convert_back : bool (default False)
        If True converts from decoded human readable values to encoded values.
    """
    if dir_path is None:
        dir_path = os.environ.get('UKBB_CACHE_DIR')

    encoding_dicts = [
        'category', 'catbrowse', 'record_column', 'field', 'recommended', 
        'fieldsum', 'encoding', 'snps', 'instances', 'publication', 'record_table',
        'returns', 'schema', 'ehierint', 'ehierstring', 'insvalue', 'esimpdate', 'esimpint',
        'esimpreal', 'esimpstring', 'esimptime',        
    ]
    cached_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    download = False
    for file in encoding_dicts:
        if f'{file}.txt' not in cached_files:
            download = True
            break
    if download:
        download_encodings(dir_path)

    series_dict = {}
    columns = df.columns
    for c in columns:
        series = column_raw_to_labeled(df, c, is_name=is_name, dir_path=dir_path, cached=True, convert_back=convert_back)
        if type(series) != type(42):
            series_dict[c] = series
        else:
            series_dict[c] = df[c].copy()

    _df = pd.concat(series_dict, axis=1)
    return _df

def get_data_codings(category_ids=[], field_ids=[]):
    """
    Returns the data codings dictionary for given fields and categories.
    Parameters:
        category_ids : list
            UK Biobank category ids in a list. Codings for all contained fields are fetched.
        field_ids:
            Single UK Biobank field ids in a list.
    """
    showcase_mapping_tsv_path = os.environ.get('SHOWCASE_MAPPING_FILE')
    f_tsv = pd.read_csv(showcase_mapping_tsv_path, sep='\t')
    f_tsv = f_tsv.loc[
        f_tsv['Category'].isin(category_ids) | f_tsv['FieldID'].isin(field_ids)
    ]
    data_codings = f_tsv.groupby('Coding')['FieldID'].apply(list).to_dict()
    return data_codings

def dask_apply_decorator(func):
    '''
    This is a decorator function intended to be used with modifiers. 
    It wraps the given modifier with the dask.dataframe.DataFrame.apply function to execute it faster.
    If you want to use this decorator your modifier definition must be a bit different then the default way.
    The following is a reference example. You must define a parameter, here called "row" for example, which must not
    be given a value in the modifier_arguments dictionary.
    
    ```python
    # defining a modifier
    from ..ukbb_data_handler import dask_apply_decorator
    @dask_apply_decorator
    def add_5(row, column_name, add=5):
        row[column_name] = row[column_name] + add
        return row
    
    # using it for fetching the data
    with_modifier = dh.fetch_ukb_main(
        field_ids=[
            874, 864, 894, 884, 914, 904
        ],
        modifier=modifiers.utils.add_5,
        modifier_arguments={
            'column_name':'864-0.0', 'add':2
        },
    )
    ```
    '''
    @functools.wraps(func)
    def wrapper(df, **kwargs):
        df = dd.from_pandas(df, npartitions=10)
        df = df.apply(partial(func, **kwargs), axis=1, meta=df)
        df = df.compute()
        return df
    return wrapper

def _parallel_modifier_runner(func, df, modifier_arguments, p=4, num_chunks=10):
    with Pool(processes=p) as pool:
        n = int(len(df) / num_chunks)
        list_df = [df[i:i+n] for i in range(0,len(df),n)]
        results = [pool.apply_async(partial(func, **modifier_arguments), (_df,)) for _df in list_df]
        result_list = []
        for r in results:
            result_list.append(r.get())
        final_df = pd.concat(result_list)
    return final_df

def fetch_ukb_main(
    category_ids=[],
    field_ids=[],
    modifier=None,
    modifier_arguments={},
    rename_fields=False,
    rename_after_modifier=False,
    tsv_name=None,
    overwrite_cache=False,
    #use_dask=False,
    num_processes=1,
    num_chunks=10,
):
    """
    Loads a subset of columns from the main UK Biobank dataset.
    
    Parameters:
        category_ids : list[int] (default [])
            List of UK Biobank category IDs. All contained fields will be fetched.
        field_ids : list[int] (default []])
            List of UK Biobank field IDs.
        modifier : function (default None)
            An optional modifier function to modify the resulting data frame. Can be any function which takes the dataframe as
            first argument and returns its modified version. Some useful modifiers for standard tasks are
            provided in the ukbb_data_handler.modifiers module.
        modifier_arguments : dict  (default {})
            Additional key word arguments for the modifier function.
        rename_fields : bool (default False)
            Changes column names form field ID to clear name. Requires the SHOWCASE_MAPPING_FILE environment variable
            to be specified. Renaming happens before potential modifier function call.
        tsv_name : str (default None)
            If specified saves the dataframe under this file name into the directory specified with the
            UKBB_CACHE_DIR environment variable.
        rename_after_modifier : bool (default False)
            Like for rename_fields, dataframe columns will be renamed from field ID to clear name, however  after
            a potential modifier function was applied. Overwrites rename_fields.
        overwrite_cache : bool (default False)
            If set to True, the requested fields will be extracted and overwrite potentially existing cached versions.
        num_processes : int (default 1)
            If specified, this many CPUs will be used to apply the modifier function to multiple chunks of
            the dataframe in parallel via multiprocessing. Important: If the modifier uses summary information, like the
            mean of a column, this argument should not be used!
        num_chunks : int (default 10)
            Number of chunks to use when using num_processes > 1. Dataframe is divided into chunks, and the modifier is
            applied to each separately and in parallel.
    """
    #showcase_mapping_tsv_path = os.environ.get('SHOWCASE_MAPPING_FILE')
    ukb_main_csv_file_path = os.environ.get('UKBB_BASKET_PATH')
    dir_path_cache = os.environ.get('UKBB_CACHE_DIR')

    if not os.path.isdir(dir_path_cache):
        log.info(f'Cache dir does not exists. Creating one named {dir_path_cache}')
        os.mkdir(dir_path_cache)
    
    assert os.path.isdir(dir_path_cache)
    assert isinstance(category_ids, list)
    assert isinstance(field_ids, list)
    for _id in category_ids:
        assert isinstance(_id, int)
    
    basket_id = ukb_main_csv_file_path.split('/')[-1].split('.')[0]
    df_mapping_parsed_fields = _get_showcase_mapping(category_ids=category_ids, field_ids=field_ids)
    parsed_fields = df_mapping_parsed_fields.FieldID.unique()
    
    def _extract_fields(df_mapping_parsed_fields, ukb_main_csv_file_path):
        # Setting dtypes in read:csv function used to implement dask loading. However, setting dtypes in dataframe
        # clashes with functions for toggling data value encoding. Thus omitted here. Dask loading not faster than pandas
        # anyway.
        #_fields = []
        #df_types = pd.read_csv(showcase_mapping_tsv_path, sep='\t')
        #_types = {
        #    'Continuous': float,
        #    'Categorical single': object, #pd.Int64Dtype(), "TypeError: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'"
        #    'Date': str,
        #    'Integer': np.float64, #np.int64, , #pd.Int64Dtype(),
        #    'Text': str,
        #    'Categorical multiple': object, #pd.Int64Dtype(),
        #    'Time': str, #?
        #    'Compound': str #?
        #}
        #df_types['ValueType_pd'] = df_types['ValueType'].apply(lambda x: _types[x])
        #for field in df_mapping_parsed_fields[['Field', 'FieldID']].values.tolist():
        #    _fields.append(field[1])

        field_ids = df_mapping_parsed_fields['FieldID'].values
        def include_all_instances(x):
            for f in field_ids:
                if x.split('-')[0] == str(f):
                    return x
        df_for_dtypes = pd.read_csv(ukb_main_csv_file_path, usecols=include_all_instances, nrows=1)
        column_names = list(df_for_dtypes.columns)
        column_names.append('eid')

        # _dtypes_for_fields = {str(k): v for k, v in zip(field_ids, df_types.loc[df_types.FieldID.isin(field_ids), 'ValueType_pd'].to_list())}
        #_dtypes = {'eid': np.int64}
        #for _c in column_names:
        #    for f in field_ids:
        #        if _c.split('-')[0] == str(f):
        #            _dtypes[str(_c)] = _dtypes_for_fields[str(f)]

        #_start = timer()
        #if use_dask:
        #    df = dd.read_csv(
        #        ukb_main_csv_file_path,
        #        usecols=column_names,
        #        dtype=_dtypes,
        #        low_memory=False,
        #        sample=int(1e6)
        #    )
        #    df = df.compute()
        #else:
        #    chunk = pd.read_csv(
        #        ukb_main_csv_file_path,
        #        usecols=column_names,
        #        dtype=_dtypes,
        #        chunksize=1000000
        #    )
        #    df = pd.concat(chunk)
        #_end = timer()
        _start = timer()
        chunk = pd.read_csv(
                ukb_main_csv_file_path,
                usecols=column_names,
                chunksize=1000000
            )
        df = pd.concat(chunk)
        _end = timer()
        log.info(f'Loaded main dataset file in {timedelta(seconds=_end-_start)}')

        withdrawn_eids = get_withdrawn_eids()
        df = df[~(df['eid'].isin(withdrawn_eids))]
        df.set_index('eid', inplace=True)
        return df

    does_all_field_csv_exists = True
    for f_id in parsed_fields:
        field_df_name = _get_cache_fname(basket_id, f_id)
        fpath = os.path.join(dir_path_cache, field_df_name)
        if not os.path.exists(fpath):
            does_all_field_csv_exists = False
            break
    
    if does_all_field_csv_exists and (not overwrite_cache):
        df = get_fields_from_cache(parsed_fields, basket_id, dir_path_cache)
    else:
        log.info(f"For at least one of the fields, the cached fields feather file does not exists in cache folder")
        df = _extract_fields(df_mapping_parsed_fields, ukb_main_csv_file_path)
        cache_fields(df, parsed_fields, basket_id, dir_path_cache)
    
    if tsv_name:
        df.to_csv(os.path.join(dir_path_cache, tsv_name), sep="\t")

    # apply optional modifier
    if rename_fields and (not rename_after_modifier):
        df = toggle_column_name(df, to_id=False)
    if modifier:
        if num_processes > 1:
            df = _parallel_modifier_runner(modifier, df, modifier_arguments, p=num_processes, num_chunks=num_chunks)
        else:
            df = modifier(df, **modifier_arguments)
    if rename_after_modifier:
        df = toggle_column_name(df, to_id=False)

    return df
