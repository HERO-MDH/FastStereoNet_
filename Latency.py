import pandas as pd
from tabulate import tabulate
from sklearn import preprocessing
from sklearn import tree
import pickle
def convert(input):


    output=[]
    for i in input:
        inX = 37
        inY = 37
        inZ = 3
        for j in i:
            if(j[0]=='conv2d'):
                print(j[0])
                output.append([inX,inY,inZ,j[1],j[3]])
                inZ=j[1]
                if(j[2]=='valid'):
                    inX=(inX-j[3])+1
                    inY = (inY - j[3]) + 1
    return output
def List_to_df(input):
    df=pd.DataFrame(input)
    return df
def Latency_estimation(arch):
    converted_list=convert(arch)
    converted_df=List_to_df(converted_list)
    scaled_df=preprocessing.scale(converted_df)
    regr = tree.DecisionTreeRegressor(max_depth=10)
    loaded_model = pickle.load(
        open('C:\\Users\\azi01\\Dropbox\\My PC (LAP-5CG9106B3R)\\Downloads\\DecisionTree.sav', 'rb'))
    result = loaded_model.predict(scaled_df)
    print(result)
    return result.sum()
if __name__=='__main__':
    init = [[['conv2d', 32, 'same', 5],
             ['conv2d', 64, 'same', 5],
             ['none', 0, 'none', 0],
             ['conv2d', 64, 'same', 5],
             ['none', 0, 'none', 0],
             ['conv2d', 64, 'same', 5],
             ['conv2d', 64, 'same', 5],
             ['none', 0, 'none', 0],
             ['conv2d', 64, 'valid', 3],
             ['conv2d', 64, 'same', 5],
             ['conv2d', 64, 'same', 5],
             ['conv2d', 32, 'valid', 35]],
            [['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['conv2d', 32, 'valid', 37]]]
    '''list1=convert(init)
    df=List_to_df(list1)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    df=preprocessing.scale(df)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    regr = tree.DecisionTreeRegressor(max_depth=10)
    loaded_model = pickle.load(
        open('C:\\Users\\azi01\\Dropbox\\My PC (LAP-5CG9106B3R)\\Downloads\\DecisionTree.sav', 'rb'))
    result = loaded_model.predict(df)
    print(result)
    print(result.sum())'''
    print(Latency_estimation(init))