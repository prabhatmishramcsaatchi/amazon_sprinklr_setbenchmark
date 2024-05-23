import json
import os
import boto3
import pandas as pd
from io import BytesIO
import io
import numpy as np





month_to_quarter = {
'January': 'Q1',
'February': 'Q1',
'March': 'Q1',
'April': 'Q2',
'May': 'Q2',
'June': 'Q2',
'July': 'Q3',
'August': 'Q3',
'September': 'Q3',
'October': 'Q4',
'November': 'Q4',
'December': 'Q4'
}


performance_metrics = [
'Engagements',
'Organic Impressions',
'Organic Reach',
'Total Impressions'

]

##Comment sentiment

bp_metrics = [
'Engagement Rate',
'Comments per Impression',
'Likes per Impression',
'Comment Sentiment'

]




paid_columns_list = [
    'Estimated Ad Recall Lift Rate',
    'Cost per Estimated Ad Recall Lift',
    'CPM',
    'CPE',
    'Total Engagements',
    'CPC',
    'CTR',
    'Cost per Thruplay',
    'Video Average Play Time'
]

# Define the average benchmark columns list
average_benchmark_columns = [
    'AverageEstimated ad recall lift rate',
    'AverageCost per estimated ad recall lift',
    'AverageCPM',
    'AverageCPE',
    'AverageTotal Engagements',
    'AverageCPC',
    'AverageCTR',
    'AverageCost per Thruplay',
    'AverageVideo average play time'
]

# Define the median benchmark columns list
median_benchmark_columns = [
    'MedianEstimated ad recall lift rate',
    'MedianCost per estimated ad recall lift',
    'MedianCPM',
    'MedianCPE',
    'MedianTotal Engagements',
    'MedianCPC',
    'MedianCTR',
    'MedianCost per Thruplay',
    'MedianVideo average play time'
]

s3 = boto3.client('s3')
def convert_to_month(date):
    try:
        return date.strftime('%B')
    except:
        return ""
        
        

def convert_month_to_quarter(month):
    return month_to_quarter.get(month)
    
    


 
def read_excel_from_s3(source_bucket, key):
    obj = s3.get_object(Bucket=source_bucket, Key=key)
    file_content=obj['Body'].read()
    read_excel_data=io.BytesIO(file_content)
    df = pd.read_excel(read_excel_data)
    return df   

def load_files(bench_mark_file):
    source_bucket = 'wikitablescrapexample'
    folder = 'amazon_sprinklr_pull/mappingandbenchmark/'
    
    bench_mark_data= read_excel_from_s3(source_bucket, folder + bench_mark_file)
    
    return  bench_mark_data   
    

    
# def process_paid_data(df):
#     df = df.drop('Objective', axis=1)
#     df = df.rename(columns={'Cleaned': 'Objective'})
#     benchmark_df = load_files('PaidBenchmarks.xlsx')
#     benchmark_df['Objective']=benchmark_df['Objective'].str.lower()
#     benchmark_df['Region']=benchmark_df['Region'].str.lower()
#     df['Objective']=df['Objective'].str.lower()
#     df['Region']=df['Region'].str.lower()
#     df = pd.merge(df, benchmark_df, on=['Objective','Region'])

#     column_map = {
#         'Estimated Ad Recall Lift Rate': ['AverageEstimated ad recall lift rate', 'MedianEstimated ad recall lift rate'],
#         'Cost per Estimated Ad Recall Lift': ['AverageCost per estimated ad recall lift', 'MedianCost per estimated ad recall lift'],
#         'CPM': ['AverageCPM', 'MedianCPM'],
#         'CPE': ['AverageCPE', 'MedianCPE'],
#         'Total Engagements': ['AverageTotal Engagements', 'MedianTotal Engagements'],
#         'CPC': ['AverageCPC', 'MedianCPC'],
#         'CTR': ['AverageCTR', 'MedianCTR'],
#         'Cost per Thruplay': ['AverageCost per Thruplay', 'MedianCost per Thruplay'],
#         'Video Average Play Time': ['AverageVideo average play time', 'MedianVideo average play time']
#     }

#     for column, [average_benchmark, median_benchmark] in column_map.items():
#         df[average_benchmark].replace(0, np.nan, inplace=True)
#         df[median_benchmark].replace(0, np.nan, inplace=True)
#         df[average_benchmark] = ((df[column] - df[average_benchmark]) / df[average_benchmark])
#         df[median_benchmark] = ((df[column] - df[median_benchmark]) / df[median_benchmark])
#         df[average_benchmark].replace(np.nan, 0, inplace=True)
#         df[median_benchmark].replace(np.nan, 0, inplace=True)
#     df['Quarter'] = np.where(df['Quarter_x'].notna(), df['Quarter_x'], df['Quarter_y'])
#     df = df.drop(columns=['Quarter_x', 'Quarter_y'])



#     return df



# Define a function to process benchmarks
def process_paid_benchmark(df, benchmark_df, benchmark_columns, benchmark_type):
    merged_df = pd.merge(df, benchmark_df, on='RegionObjective', how='left')
    for paid_column, benchmark_column in zip(paid_columns_list, benchmark_columns):
        final_column_name = "Benchmark" + paid_column
       
        merged_df[final_column_name] = ((merged_df[paid_column] - merged_df[benchmark_column]) / merged_df[benchmark_column])
        merged_df[final_column_name] = merged_df[final_column_name].replace([np.inf, -np.inf], np.nan)
    merged_df['Benchmark Type'] = benchmark_type
    
    return merged_df.drop(columns=benchmark_columns)


# Updated process_paid_data function
def process_paid_data(df):
    df = df.drop('Objective', axis=1)
    df['RegionObjective'] = (df['Country'].str.lower() + df['Cleaned'].str.lower()).str.replace(' ', '')
    
    median_benchmark_df = load_files('median_paid.xlsx')
    average_benchmark_df = load_files('average_paid.xlsx')
    
    median_benchmark_df.drop_duplicates(subset='RegionObjective', inplace=True)
    average_benchmark_df.drop_duplicates(subset='RegionObjective', inplace=True)
    
    median_benchmark_df['RegionObjective'] = median_benchmark_df['RegionObjective'].str.lower().str.replace(' ', '')
    average_benchmark_df['RegionObjective'] = average_benchmark_df['RegionObjective'].str.lower().str.replace(' ', '')
    
    # Drop 'Region' column from benchmark dataframes
    median_benchmark_df = median_benchmark_df.drop(columns=['Region'])
    average_benchmark_df = average_benchmark_df.drop(columns=['Region'])
    
    
    final_median_df = process_paid_benchmark(df, median_benchmark_df, median_benchmark_columns, 'Median')
    final_average_df = process_paid_benchmark(df, average_benchmark_df, average_benchmark_columns, 'Average')
    
    final_df = pd.concat([final_median_df, final_average_df], axis=0).sort_index(kind='merge')
    # Create a new 'quarter' column that takes non-null values from 'quarter_x' and 'quarter_y'
    final_df['Quarter'] = final_df['Quarter_x'].where(final_df['Quarter_x'].notna(), final_df['Quarter_y'])
    
    # Drop 'quarter_x' and 'quarter_y' columns
    final_df = final_df.drop(columns=['Quarter_x', 'Quarter_y','Count'])
    
    return final_df



def generate_matcher(df):
    """Generate the Matcher column for a given dataframe."""
    
    # Check if 'Delivery' column is 'Boosted' and set quarter to 'all' accordingly
    df['Quarter_for_Matcher'] = np.where(df['Delivery'].str.lower() == 'boosted', 'all', df['Quarter'])

    df['Matcher'] = (df['Country'] + df['Platform'] + df['Delivery'] + df['Quarter_for_Matcher']).str.lower()

    # Drop the 'Quarter_for_Matcher' column as it is no longer needed
    df.drop('Quarter_for_Matcher', axis=1, inplace=True)
    
    return df




def process_organic_data(df):
    df = generate_matcher(df)
    
    benchmarks_median = load_files('median_benchmarks.xlsx')
    benchmarks_mean = load_files('mean_benchmarks.xlsx')
    
    benchmarks_median = generate_matcher(benchmarks_median)
    benchmarks_mean = generate_matcher(benchmarks_mean)
    
    df_median = calculate_performance_for_benchmark(df.copy(), benchmarks_median, typeof='Median')
    df_mean = calculate_performance_for_benchmark(df.copy(), benchmarks_mean, typeof='Average')
    
    df = pd.concat([df_median, df_mean])
    return df
    
    
def calculate_performance_metric(row, benchmark_row, metric):
    """Calculate performance metric for a given row."""
    if benchmark_row.empty or np.isnan(benchmark_row[metric].values[0]):
        return np.nan
    else:
        benchmark = benchmark_row[metric].values[0]
        # Check if the benchmark is zero to avoid division by zero
        if benchmark == 0:
            return np.nan
        else:
            return (row[metric] - benchmark) / benchmark


def calculate_bp_metric(row, benchmark_row, metric):
    """Calculate bp metric for a given row."""
    if benchmark_row.empty or np.isnan(benchmark_row[metric].values[0]):
        return np.nan
    else:
        benchmark = benchmark_row[metric].values[0]
        # Check if both benchmark and row[metric] are zero
        if benchmark == 0 and row[metric] == 0:
            return np.nan
        else:
            return 10000 * (row[metric] - benchmark)
        
        
        
# def fetch_benchmark_for_row(row, benchmarks):
#     """Fetch the benchmark for a given row."""
#     benchmark_row = benchmarks.loc[benchmarks['Matcher'] == row['Matcher']]
    
#     # Use annual data if benchmark sample for a quarter is too small and delivery is organic
#     if benchmark_row['Count'].values < 10 and 'organic' in row['Matcher']:
#         benchmark_row = benchmarks.loc[benchmarks['Matcher'] == row['Matcher'][0:-2] + 'all']
        
#     return benchmark_row

# def fetch_benchmark_for_row(row, benchmarks):
    
#     """Fetch the benchmark for a given row."""
#     benchmark_row = benchmarks.loc[benchmarks['Matcher'] == row['Matcher']]
    
#     # Use annual data if benchmark sample for a quarter is too small and delivery is organic
#     if benchmark_row.empty or (benchmark_row['Count'].values < 10 and 'organic' in row['Matcher']):
#         modified_matcher = row['Matcher'][0:-2] + 'all'
#         benchmark_row = benchmarks.loc[benchmarks['Matcher'] == modified_matcher]
        
#     return benchmark_row
def fetch_benchmark_for_row(row, benchmarks):
    """Fetch the benchmark for a given row."""
    benchmark_row = benchmarks.loc[benchmarks['Matcher'] == row['Matcher']]
    modified_matcher = row['Matcher']
    
    # Check if 'Matcher' is a string and if benchmark_row is not empty or meets the condition
    if isinstance(row['Matcher'], str) and (benchmark_row.empty or (benchmark_row['Count'].values[0] < 10 and 'organic' in row['Matcher'])):
        modified_matcher = row['Matcher'][0:-2] + 'all'
        benchmark_row = benchmarks.loc[benchmarks['Matcher'] == modified_matcher]
        
    return benchmark_row, modified_matcher

def calculate_performance_for_benchmark(df, benchmarks, typeof):
    """Calculate performance metrics against given benchmarks."""
    
    df['Benchmark Type'] = typeof
    
    for index, row in df.iterrows():
        benchmark_row, modified_matcher = fetch_benchmark_for_row(row, benchmarks)
        
        # Update the 'Matcher' column in the original DataFrame
        df.at[index, 'Matcher'] = modified_matcher
        
        for metric in performance_metrics:
            df.at[index, metric + ' vs Benchmark'] = calculate_performance_metric(row, benchmark_row, metric)
        
        for metric in bp_metrics:
            df.at[index, metric + ' vs Benchmark(BPS)'] = calculate_bp_metric(row, benchmark_row, metric)
    
    return df

  
 
        # Define a function to select and return the appropriate columns
def select_columns(row):
    if row['is_paiddata'] == 0:
        return pd.Series([row['Account'], row['Platform'], None, None])
    else:  # Assuming 'is_paiddata' == 1
        return pd.Series([None, None, row['AD_ACCOUNT'], row['CHANNEL']])
      
def lambda_handler(event, context):
    try:
        bucket_name = 'wikitablescrapexample'
        key_event=event['Records'][0]['s3']['object']['key']
    
        # Read data from S3
        #key_event='amazon_sprinklr_pull/finalmaster/cleaned_master_tab_2024-01-10_2024-02-24.csv'
        obj = s3.get_object(Bucket=bucket_name, Key=key_event)
        df3 = pd.read_csv(BytesIO(obj['Body'].read()))

        df3['Date'] = pd.to_datetime(df3['Published Date'].apply(lambda x: str(x).split(' ')[0]))
        df3['Month'] = df3['Date'].apply(lambda x: convert_to_month(x))
        df3['Quarter'] = df3['Month'].apply(lambda x: convert_month_to_quarter(x))
       
    
        min_pull_date = df3['Pull Date'].min()
        max_pull_date = df3['Pull Date'].max()

        paid_df = df3[df3['is_paiddata'] == 1]
        organic_df = df3[df3['is_paiddata'] == 0]

        # Process organic/boosted dataframe
        processed_organic_df = process_organic_data(organic_df)
        
    
        # Process paid dataframe
        processed_paid_df = process_paid_data(paid_df)
        
        
        # Identify rows where Quarter is 'All' or NaN or an empty string
        mask = (processed_paid_df['Quarter'] == 'All') | (processed_paid_df['Quarter'].isna()) | (processed_paid_df['Quarter'] == '')

        
        # Extract the month and quarter from the 'Pull Date' column
        processed_paid_df.loc[mask, 'Month'] = processed_paid_df[mask]['Pull Date'].apply(lambda x: convert_to_month(pd.to_datetime(x, format='%Y-%m-%d')))
        processed_paid_df.loc[mask, 'Quarter'] = processed_paid_df[mask]['Month'].apply(convert_month_to_quarter)
        
        # Update the 'Date' column to be equal to 'Pull Date'
        processed_paid_df.loc[mask, 'Date'] = processed_paid_df[mask]['Pull Date'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
        
        # Drop rows where 'SPENT__USD__IN_USD__SUM' is 0
        processed_paid_df = processed_paid_df[processed_paid_df['SPENT__USD__IN_USD__SUM'] != 0]

        # Concatenate processed dataframes
        processed_df = pd.concat([processed_organic_df, processed_paid_df])
        


        
        
        processed_df['Year'] = processed_df['Date'].dt.year
        

        
                # Rename columns
        processed_df.rename(columns={
            'Paid Impressions': 'Paid Impressions_1st_party',
            'IMPRESSIONS__SUM': 'Paid Impressions'
        }, inplace=True)
        
 
        # Process only rows where 'Country' is null
        df_null_country = processed_df[processed_df['Country'].isna()]
  
        if not df_null_country.empty:
        # Apply the function to each row
            selected_columns = df_null_country.apply(select_columns, axis=1)
            selected_columns.columns = ['Account', 'Platform', 'AD_ACCOUNT', 'CHANNEL']
                           
            # Save the selected_columns DataFrame to a CSV file
            csv_buffer = BytesIO()
            selected_columns.to_csv(csv_buffer, index=False)
            output_key = f'amazon_sprinklr_pull/unmatch_mapping/processed_country_null_data.csv'
            s3.put_object(Bucket=bucket_name, Key=output_key, Body=csv_buffer.getvalue())
        
        # Drop rows where 'Country' is null or empty
        processed_df = processed_df.dropna(subset=['Country'])
        processed_df = processed_df[processed_df['Country'] != '']
         
        # Save the result to S3
        csv_buffer = BytesIO()
        processed_df.to_csv(csv_buffer, index=False)
         
        key = f'amazon_sprinklr_pull/tableau_layer/masterbenchmarks_{min_pull_date}_{max_pull_date}.csv'
        s3.put_object(Bucket=bucket_name, Key=key, Body=csv_buffer.getvalue())
 
        return {
            'statusCode': 200,
            'body': 'Success! Check the output in your S3 bucket'
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': f'Error: {e}'
        }







