import pandas as pd
import os
import io
import numpy as np

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
    

def comment_header(file, line_no = 0, comments='#', delim=' '):

    '''

        TODO: Address perfomance issues!!!! Compare peformance to simply calling read.csv but ignoring the # somehow
        TODO: Maybe for i,line in enumerate(readlines) , if line == line_no is better 

        Grab a header at specified line that has a specified comment in front and split based on delim. assumes line is in the first few so first chunk is loaded and lines read

        Parameters:
            file : str 
                    Name of file to extract header from
            line_no : int
                    Line number for the header. First line is zero. 
                    Default : 0

            delim : str
                    Seperator for the headers 
                    Default : ' '
            comments : str
                    Comment markers
                    Default : '#'

        Returns:
            head : list
                    List of strings containing header names

    '''

    with open(file) as stream:
        first_line = stream.readline()
        if line_no==0:
            head=first_line
        else:
            buff = io.DEFAULT_BUFFER_SIZE
            line_size = len(first_line)# get the size of the first line, to estimate how much of a buffer to give approximately
            stream.seek(0) # go back to beginning
            #my weird approach
            try:
                head = stream.readlines(buff+(line_size)*(line_no))[line_no] #read the nth line # reverses at least the default buffer + line size per line
            except IndexError:
                #if out of range, make buffer bigger!
                stream.seek(0)
                head = stream.readlines(buff+(buff+line_size)*(line_no))[line_no]

        #simpler approach?

        # for i,line in enumerate(stream):
        #     if i == line_no:
        #         head = line
        


    # strip trailing wspace and then split into a list
    head_list = head.strip().split(delim) # split the header line at the desired delimiter
    trimmed = [i for i in head_list if i not in comments]

    return trimmed

def new_cols(inpdf,colnames):
    for col in colnames:
        if col not in inpdf.columns:
            inpdf[col] = np.nan # create an empty column with this name
    return inpdf

def raise_columns(obj):
    '''Apply to an object with a .data attribute, that is a data frame. Grab the columns and turn into a setter/getter'''

    for col in obj.data.columns:
        # has to be set to a method of the class
        setattr(obj.__class__, col, raise_col(obj,col))

def raise_col(obj,col_name):
    '''Requires .data attribute'''

    def get_col(self):
        
        return self.data[col_name]

    def set_col(self,value):

        self.data[col_name] = value

    col_prop = property(get_col,set_col)
    
    #setattr(obj, col_name,col_prop)

    return col_prop



def rebin(df,binning_coord,bw=0.25,bins = None,mode = 'average'):
   
    ''' 
        Rebin the data based on coordinates for a given new bin width.
        Default is to perform averages over these bins.
        Could also add weightings for these averages

        coord = Column name of the coordinate to create the bins from

        inplace : bool  
            if True, overwrites the .data method, else just returns the data frame.

    '''

    '''
        Create bins of a coordinate and then group by this coordinate
        - average by this group
    '''

    coord = binning_coord
        
    coord_max = df[coord].max()
    coord_min = df[coord].min() # finding min and max so it works with gfolded data


    if not bins:
        # if bin edges not explicitly provided, calculate them from bw and max and min of coord
        n_bins = int((coord_max-coord_min) // bw) #double divide is floor division
        bins = pd.cut(df[coord],n_bins)


    df_grouped = df.groupby(bins,as_index=False) # don't want another column also called coord!!

    if mode == 'average' or mode == 'mean':
        df_binned = df_grouped.mean()
    else:
        df_binned = df_grouped.sum()

    return df_binned



if __name__ == "__main__":
    

    dict1 = {"Time":[1,2,3,4],'data_a':['a','b','c','d']}

    dict2 = {"Time":[1,69,124,4,52,23],'data_b':['a','b','c','d','e','g']}

    df1 = pd.DataFrame(dict1)
    df2 = pd.DataFrame(dict2)


    # df_join = pd.concat([df1,df2],axis=1)

    find_dupes(df1,df2)

    df_merge = merge_no_dupes(df1,df2)

    print(df_merge)


    # testing comment_header

    head = comment_header('../../test_files/temp.profile', line_no= 2, comments='#', delim=' ')

    print(head)