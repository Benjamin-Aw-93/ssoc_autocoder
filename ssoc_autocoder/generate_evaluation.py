import pandas as pd

#takes in a df

#calling this function generates everything you need
# let n be min number of samples to be considered for weighted accuracy
def generate(df, n):
    res ={"1D accuracy":0,"2D accuracy":0,"3D accuracy":0,"4D accuracy":0,"5D accuracy":0,"Top 5 accuracy":0,"Top 3 accuracy":0,"Top 1 accuracy":0,"Weighted Accuracy":0,"Recall of selected SSOCS":0,"Precision of selected SSOCS":0}
    
    #this prints the 1D to 5D accuracy============================================================
    key_list=list(res.keys())

    for n in range(1, 6):
        similarity_percentage = calculate_similarity_percentage(df['pred1'], df['SSOC'], n)
        res[key_list[n-1]] =round(similarity_percentage,2)

    #get top 1,3,5 accuracy=======================================================================
    res = top_5_accuracy(df, res)

    # get weighted accuracy=======================================================================
    res = weighted_accuracy(df, n,res)

    #get precision and recall of selected SSOCs
    res = selected_ssocs_metrics(df, res)

    return res

def calculate_similarity_percentage(s1, s2, n):
    s1_str = s1.astype(str)
    s2_str = s2.astype(str)
    return sum(s1_str.str[:n] == s2_str.str[:n]) / len(s1) * 100


def top_5_accuracy(df, res):
        pred_columns = ['pred1', 'pred2', 'pred3', 'pred4', 'pred5']
        total_rows = len(df)
        matching_rows = 0

        for col in pred_columns:
            matching_rows += sum(df[col] == df['SSOC'])

        percentage = (matching_rows / total_rows) * 100
        res['Top 5 accuracy'] = round(percentage,2)

        pred_columns = ['pred1', 'pred2', 'pred3']

        total_rows = len(df)
        matching_rows = 0

        for col in pred_columns:
            matching_rows += sum(df[col] == df['SSOC'])

        percentage = (matching_rows / total_rows) * 100
        res['Top 3 accuracy'] = round(percentage,2)

        pred_columns = ['pred1']

        total_rows = len(df)
        matching_rows = 0

        for col in pred_columns:
            matching_rows += sum(df[col] == df['SSOC'])

        percentage = (matching_rows / total_rows) * 100
        res['Top 1 accuracy'] = round(percentage,2)

        return res


def weighted_accuracy(df, n, res):
    list_of_ssocs = df['SSOC'].unique().tolist()
    list_of_ssocs.sort()
    total = len(list_of_ssocs)

    df['correct'] = df['SSOC'] == df['pred1']
    ssoc_counts = df['SSOC'].value_counts()
    valid_ssocs = ssoc_counts[ssoc_counts > n]
    ssoc_accuracies = df.groupby('SSOC')['correct'].mean()
    valid_ssoc_accuracies = ssoc_accuracies[valid_ssocs.index]
    

    weighted = sum(valid_ssoc_accuracies) / len(valid_ssoc_accuracies)
    res['Weighted Accuracy'] = round(weighted,2)

    
    return res


def selected_ssocs_metrics(df, res):
    ssoc_value_counts = df['SSOC'].value_counts()
    filtered_df = df[df['SSOC'].isin(ssoc_value_counts[(ssoc_value_counts>=20) & (ssoc_value_counts<=35)].index)]
    filtered_ssocs = list(set(filtered_df['SSOC'].tolist()))
    filtered_ssocs.sort()

    recall_values = {}
    for ssoc in filtered_ssocs:
        tp = (df[df['SSOC']==ssoc]['SSOC']==df[df['SSOC']==ssoc]['pred1']).sum()
        total = (df[df['SSOC']==ssoc]['SSOC']==df[df['SSOC']==ssoc]['pred1']).count()
        recall = tp/total
        recall_values[ssoc] = round(recall,2)
    

    res['Recall of selected SSOCS'] = round(sum(recall_values.values())/len(recall_values),2)

    precision_values = {}
    for ssoc in filtered_ssocs:
        tp = (df[df['SSOC']==ssoc]['SSOC']==df[df['SSOC']==ssoc]['pred1']).sum()
        fp = (df[df['pred1']==ssoc]['SSOC']!=ssoc).sum()
        precision = tp/(tp+fp)
        precision_values[ssoc] = round(precision,2)
        
    res['Precision of selected SSOCS'] = round(sum(precision_values.values())/len(precision_values),2)


    return res




df = pd.read_csv('yeesen_res.csv')
results = generate(df,5)
print(results)