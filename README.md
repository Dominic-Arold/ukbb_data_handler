# ukbb_data_handler

This package enables to conveniently work with UK BioBank data by streamlining data loading 
and standard data engineering. Its features are born from the needs in everyday scientific practice.
Specifically, ```ukbb_data_handler``` provides:

- automated parsing of the main dataset file to cache specific fields or whole categories for quick access.
- "modifier" functionality to make operations on the raw fetched data in a controlled way. Different common and specific dataframe manipulations are included in the modifiers module
- functions to conveniently toggle between encoded and decoded "human readable" column names and data values

## Installation and Setup

### 1. Installing the Package

**Install with poetry**

Add the following to your projects pyproject.toml file in the [tool.poetry.dependencies] section:
```toml
ukbb-data-handler = {git = "https://github.com/Dominic-Arold/ukbb_data_handler.git", tag = "0.2.0"}
```
And update your environment in the shell:
```bash
poetry update
```

**Install with pip**

```bash
pip install git+https://github.com/Dominic-Arold/ukbb_data_handler.git@0.2.0
```


### 2. Downloading Showcase Mapping File

For the package to work correctly an up to date "Data Dictionary of Showcase fields"
file is required. Download the version in tsv format from the [UK Biobank webpage](https://biobank.ctsu.ox.ac.uk/crystal/exinfo.cgi?src=AccessingData) 
and safe it anywhere on your system (named below "Data_Dictionary_Showcase.tsv").

### 3. Setting the Environment Variables

To be able to read and manipulate UK Biobank data, we need to set 4 environment variables:

- **SHOWCASE_MAPPING_FILE**: Path to the Showcase Mapping File downloaded above.
- **UKBB_CACHE_DIR**: Path to the cache directory. Parsed fields from the main dataset will be saved here to be accessed quickly.
- **UKBB_BASKET_PATH**: Path to the UK Biobank main dataset (unpacked csv file).
- **PARTICIPANT_WITHDRAWALS_DIR**: Path to directory which contains csv files of participant EIDs who withdrew from the study. Those participants will be excluded automatically when fetching data.


On Linux, these environment variables can be set globally to be used in any project e.g. in the ***.bashrc*** file.
Otherwise, one can set up  the environment variables directly in the python analysis script:

```python
import os

UNAME = os.environ.get('USERNAME')
BASKET_DIR = "/.../..." # directory path to your basket files
os.environ['UKBB_BASKET_PATH'] = os.path.join(BASKET_DIR, "ukbXXXXXX.csv")
os.environ['SHOWCASE_MAPPING_FILE'] = os.path.join(BASKET_DIR, "Data_Dictionary_Showcase.tsv")
os.environ['UKBB_CACHE_DIR'] = f"/home/{UNAME}/ukbb_cache/"
os.environ['PARTICIPANT_WITHDRAWALS_DIR'] = os.path.join(BASKET_DIR, 'participant_withdrawals')
```

## Use Cases

Documentation for any function is available by running the help function in python, for example:

```python
from ukbb_data_handler import ukbb_data_handler as dh
help(dh.fetch_ukb_main)
```

### Parsing UK Biobank data

The **fetch_ukb_main** function allows to easily parse whole categories or single fields of the main dataset. Just pass
any field or category ID contained in your Basket, see 
[UK Biobank Showcase](https://biobank.ctsu.ox.ac.uk/crystal/browse.cgi?id=-2&cd=essential_info).
Once loaded for the first time, the fetched table columns get cached for quick access on the next call of the function 
to avoid time-consuming parsing of the large main dataset. All instances of parsed fields are loaded. The *rename_fields*
keyword converts column names from field ID to clear name. Run "help(fetch_ukb_main)" for documentation of additional 
options.

```python
from ukbb_data_handler import ukbb_data_handler as dh

df = dh.fetch_ukb_main(
    category_ids=[
        100094, # Population Characteristics
    ],
    field_ids=[
        54, # UK Biobank assessment centre
        53, # Date of attending assessment centre
    ],
    rename_fields=True
)
```


### Decoding/encoding column names and data values

By default, column names are loaded as encoded field IDs and data as encoded values. Column names and data values can be
toggled to clear names/values anytime:

```python
# Convert column names from Field IDs to clear names. Use to_id=True to reverse
df = dh.toggle_column_name(df, to_id=False)
```

Similarly, we can toggle data values from encoded to readable decoded values.
This can be done for single columns or for the complete dataframe.

```python
df = dh.fetch_ukb_main(
    field_ids=[
        1980, # Worrier / anxious feelings
    ],
    rename_fields=True,
)

print(df['Worrier / anxious feelings-0.0'].unique())
# output:
# [ 1.  0. -1. -3. nan]
df['Worrier / anxious feelings-0.0'] = dh.column_raw_to_labeled(df, 'Worrier / anxious feelings-0.0', is_name=True)
print(df['Worrier / anxious feelings-0.0'].unique())
# output:
# ['Yes' 'No' 'Do not know' 'Prefer not to answer' nan]
```

When fetching data, we set rename_fields=True to get clear names in column names, just like with toggle_column_name.
Thus, df now has the column name "Worrier / anxious feelings". The column_raw_to_labeled function then converted data 
values from encoded integers to decoded values. There, the keyword is_name=True was needed, because the column 
name now was in clear name. If we had loaded more fields and wanted to decode data from all columns we 
could have instead run 

```python
df = dh.fetch_ukb_main(
    field_ids=[
        20126, # Bipolar and major depression status
        1980, # Worrier / anxious feelings
    ],
)
df = dh.toggle_encoding(df)
```

### Modifiers

A modifier is just any function which takes as first argument a dataframe and returns a modified version of it.
This way, data manipulations are compartmentalized for ease of use. Some modifiers for standard tasks are included in 
the ```modifiers``` submodule. Below is an example usage of the ```age_at``` modifier, which adds a column containing 
the age of each participant for the first assessment visit (instance 0):

```python
from ukbb_data_handler import modifiers
df = dh.fetch_ukb_main(
    field_ids = [
      # required fields for age_at modifier
      52, # Month of birth
      34, # Year of birth
      53 # Date of attending assessment centre	
    ],
    # run the age_at modifier directly after data loading by specifying these args:
    modifier = modifiers.age_modifiers.age_at, 
    modifier_arguments = {'field' : 53, 'instance' : 0}
)
# Alternatively, the same is achieved by applying the modifier function anytime by hand:
# df = modifiers.age_modifiers.age_at(df, field = 53, instance = 0)
# New columns contains age at first assessment visit
print(df['Age_at_53-0.0'].mean())
```

The ```filter_instances``` modifier, as another example, keeps only the columns for each field 
which belong to a specific assessment visit (instance). Columns for any other instance get dropped:

```python
df = dh.fetch_ukb_main(
    field_ids = [52, 34, 53],
    modifier = modifiers.filter.filter_instances, 
    modifier_arguments = {
        'keep_instances': [0]
    }
    
)
print(df.columns.to_list())
# Output
# ['34-0.0', '52-0.0', '53-0.0']
```

The ```sequential``` modifier runs multiple modifiers one after another. Its *modifiers* argument is a structured list. 
Each element specifies a modifier and its arguments. For example, we might want to work only on data for 
the imaging visit (instance 2) and need to know the participants' age at this assessment:

```python
df = dh.fetch_ukb_main(
    field_ids = [52, 34, 53],
    modifier = modifiers.sequential.sequential, 
    modifier_arguments = {
        'modifiers': [
            # first remove all unneeded instances, i.e. assessment visits
            {
                'modifier': modifiers.filter.filter_instances,
                'keep_instances' : [2], 
                # fields 52 and 34 only have instance 0 -> exclude them from filtering 
                'exceptions' : [52, 34]
            },
            # then compute age at imaging visit (instance 2)
            {
                'modifier': modifiers.age_modifiers.age_at,
                'field' : 53, 'instance' : 2
            },
        ],
    },
)
print(df['Age_at_53-2.0'].mean())
```

## Executing Modifiers with multiprocessing

We offer an option to run a modifier in parallel for faster execution. The dataframe is processed in chunks by multiple
threads and the results are concatenated afterwards to the result dataframe. 
Set the number of CPU cores with the *num_processes* argument and the number of chunks with *num_chunks* argument.

**Important:** If the modifier does not work "row-wise", i.e. computes something for each participant independently, 
multiprocessing can not be used then!

```python
multiprocess_df = dh.fetch_ukb_main(
    field_ids=[874, 864, 894, 884, 914, 904],
    modifier=modifiers.lifestyle_environment.IPAQ_MET_variables,
    modifier_arguments={'instance': 2},
    # use 4 CPU cores to process all chunks of size 1000 of the input dataframe
    num_processes=4,
    num_chunks=1000,
)
```