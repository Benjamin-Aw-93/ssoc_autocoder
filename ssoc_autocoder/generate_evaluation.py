import pandas as pd

#takes in a df
def generate(df):

    #this prints the 1D to 5D accuracy
    for n in range(1, 6):
        similarity_percentage = calculate_similarity_percentage(df['pred1'], df['SSOC'], n)
        print(f"{n}D Accuracy: {similarity_percentage:.2f}%")
    #get top 1,3,5 accuracy

    top_5_accuracy(df)
    # get weighted accuracy
    print(weighted_accuracy(df))

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




def weighted_accuracy(df):
    unique_classes = df['SSOC'].unique()
    class_accuracies={}
    for class_label in unique_classes:
        correct_predictions = ((df['SSOC']==class_label) & (df['pred1']==class_label)).sum()
        total_instances =  (df['SSOC']==class_label).sum()
        class_accuracy = correct_predictions / total_instances if total_instances>0 else 0
        class_accuracies[class_label] = class_accuracy    

    weighted_accuracy = sum(class_accuracy*(df['SSOC'].value_counts()[class_label]/len(df))
                                            for class_label, class_accuracy in class_accuracies.items())
    
    return weighted_accuracy

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

