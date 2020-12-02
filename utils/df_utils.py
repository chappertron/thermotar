import pandas as pd
from collections import Counter


def find_dupes(*dfs:pd.DataFrame):

    '''list of all columns in all dfs. Flattened into one list.'''
    cols = [col for df in dfs for col in df.columns.tolist()]
    unique_cols = set(cols)
    
    seen_cols = set()
    duped_cols = list()
    if len(unique_cols) < len(cols):
        for i in cols:
            if i not in seen_cols:
                seen_cols.add(i)
            else:
                duped_cols.append(i)   
            


    return duped_cols

def merge_no_dupes(*dfs):
    '''Find duplicated columns and sets said columns to be the index,
    these dfs are then concated and the index reset.'''

    duped_cols = find_dupes(*dfs)
    #only attempts to asign index if duped cols found
    if len(duped_cols)>0:
        [df.set_index(duped_cols, inplace = True) for df in dfs]
    
    return pd.concat(dfs,axis=1).reset_index()
    


if __name__ == "__main__":
    

    dict1 = {"Time":[1,2,3,4],'data_a':['a','b','c','d']}

    dict2 = {"Time":[1,69,124,4,52,23],'data_b':['a','b','c','d','e','g']}

    df1 = pd.DataFrame(dict1)
    df2 = pd.DataFrame(dict2)


    # df_join = pd.concat([df1,df2],axis=1)

    find_dupes(df1,df2)

    df_merge = merge_no_dupes(df1,df2)

    print(df_merge)
