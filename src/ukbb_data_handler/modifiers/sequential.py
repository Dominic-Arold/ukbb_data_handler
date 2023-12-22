def sequential(df, modifiers):
    """
    A modifier to run multiple modifiers in on .fetch_ukb_main call.
    Modifiers are run sequentially. Written order will be preserved during computation.
    New columns/rows created by modifiers can be used in the following modifiers.
    It takes only one modifier argument which is the following
        - modifiers (list)
            - it is list of python dict 
                - Every dict represents one modifier see below for an example
    Parameters:
        modifiers : list[dict]
            Each list entry specifies a modifier with a dictionary:
            {'modifier' : <modifier function>, <modifier arg 1> : <modifier arg value 1>, ...}
    Example usage: Get age and corresponding age groups (for first assessment visit, instance 1)

    import ukbb_data_handler.modifiers as modifiers
    df = dh.fetch_ukb_main(
        field_ids = [34, 52, 53],
        modifier=modifiers.sequential_modifier.sequential, 
        modifier_arguments={
            'modifiers': [
                {
                    'modifier': modifiers.age_modifiers.age_at,
                    'field': 53, 'instance': 1
                },
                {
                    'modifier': modifiers.age_modifiers.age_group,
                    'age_column': 'Age_at_53-1.0'
                },
            ],
        },
    )
    """

    for mod_dict in modifiers:
        mod = mod_dict.pop('modifier')
        df = mod(df, **mod_dict)
    return df
    