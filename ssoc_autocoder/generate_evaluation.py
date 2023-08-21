import pandas as pd

#takes in a df

#calling this function generates everything you need
# let n be min number of samples to be considered for weighted accuracy
def generate(df, n):
    #this prints the 1D to 5D accuracy
    for n in range(1, 6):
        similarity_percentage = calculate_similarity_percentage(df['pred1'], df['SSOC'], n)
        print(f"{n}D Accuracy: {similarity_percentage:.2f}%")
    #get top 1,3,5 accuracy

    top_5_accuracy(df)
    # get weighted accuracy
    print(weighted_accuracy(df, n))

    #get precision and recall of selected SSOCs
    selected_ssocs_metrics()

def calculate_similarity_percentage(s1, s2, n):
    s1_str = s1.astype(str)
    s2_str = s2.astype(str)
    return sum(s1_str.str[:n] == s2_str.str[:n]) / len(s1) * 100


def top_5_accuracy(df):
        pred_columns = ['pred1', 'pred2', 'pred3', 'pred4', 'pred5']
        total_rows = len(df)
        matching_rows = 0

        for col in pred_columns:
            matching_rows += sum(df[col] == df['SSOC'])

        percentage = (matching_rows / total_rows) * 100
        print(f"Top 5 Accuracy: {percentage:.2f}%")

        pred_columns = ['pred1', 'pred2', 'pred3']

        total_rows = len(df)
        matching_rows = 0

        for col in pred_columns:
            matching_rows += sum(df[col] == df['SSOC'])

        percentage = (matching_rows / total_rows) * 100
        print(f"Top 3 Accuracy: {percentage:.2f}%")

        pred_columns = ['pred1']

        total_rows = len(df)
        matching_rows = 0

        for col in pred_columns:
            matching_rows += sum(df[col] == df['SSOC'])

        percentage = (matching_rows / total_rows) * 100
        print(f"Top 1 Accuracy: {percentage:.2f}%")




def weighted_accuracy(df, n):
    list_of_ssocs = df['SSOC'].unique().tolist()
    list_of_ssocs.sort()
    total = len(list_of_ssocs)

    df['correct'] = df['SSOC'] == df['pred1']
    ssoc_counts = df['SSOC'].value_counts()
    valid_ssocs = ssoc_counts[ssoc_counts > n]
    ssoc_accuracies = df.groupby('SSOC')['correct'].mean()
    valid_ssoc_accuracies = ssoc_accuracies[valid_ssocs.index]
    print("Weighted Accuracy of SSOC predictions:")
    weighted = sum(valid_ssoc_accuracies) / len(valid_ssoc_accuracies)

    
    return round(weighted,4)


def selected_ssocs_metrics(df):
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
    print("Recall of selected SSOCS")
    print(recall_values)
    print("Precision of selected SSOCS")
    precision_values = {}
    for ssoc in filtered_ssocs:
        tp = (df[df['SSOC']==ssoc]['SSOC']==df[df['SSOC']==ssoc]['pred1']).sum()
        fp = (df[df['pred1']==ssoc]['SSOC']!=ssoc).sum()
        precision = tp/(tp+fp)
        precision_values[ssoc] = round(precision,2)
        
    print(precision_values)
