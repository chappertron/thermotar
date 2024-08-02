import pandas as pd


class Replicator:
    def __init__(
        self, objects, indexes=None, index_names=None, get_methods="inheritable"
    ):
        """Grabs list of dataframes or objects with .data methods. Concatanates them into one dataframe and sets to this dfs df"""

        if not indexes:
            indexes = range(len(objects))

        self.index_names = index_names

        self.replicas = {index: item for index, item in zip(indexes, objects)}
        # assume homogeneous
        try:
            # to do add some sort of index naming
            df = pd.concat(
                [member.data for member in objects],
                keys=indexes,
                ignore_index=False,
                join="outer",
                names=index_names,
            )
        except AttributeError:
            df = pd.concat(
                objects,
                keys=indexes,
                ignore_index=False,
                join="outer",
                names=index_names,
            )

        self.data = df

        self.objects = objects

    def updata(self):
        """Update this object's big dataframe"""

        try:
            # to do add some sort of index naming
            df = pd.concat(
                [member.data for member in self.replicas.values()],
                keys=self.replicas,
                ignore_index=False,
                join="outer",
                names=self.index_names,
            )
        except AttributeError:
            df = pd.concat(
                self.replicas.values(),
                keys=self.replicas,
                ignore_index=False,
                join="outer",
                names=self.index_names,
            )

        self.data = df

        # if get_methods == 'inheritable':
        #     # some hackery to make the method that can be used by the replicatoron
        #     for method_name in objects[0]._inheritable:

        #         print(method_name)
        #         inherited_method = getattr(objects[0],method_name)

        #         #decorate(apply_sub_df)
        #         decorated = apply_sub_df(inherited_method)
        #         #set the doc string to the docstring of the inherited method.
        #         decorated.__doc__ = inherited_method.__doc__
        #         setattr(self,method_name,decorated)

        #     # except AttributeError:
        #     #     Warning('No inheritable methods')


# a decorator for applying methods to sub frame of self.data
# def apply_sub_df(function):
#     def wrapper(self, temp, rep,**kwargs):
#         # df = self.data
#         # sub_df = df.loc[temp,rep]
#         # sub_df = function(sub_df,**kwargs)
#         # # new columns
#         # missing_cols = list(set(sub_df.columns) - set(df.columns)  )
#         # df = df_utils.new_cols(df,missing_cols)
#         # df.loc[temp,rep]=sub_df.values
#         # #return df

#         return (self,temp,rep)

#     return wrapper
