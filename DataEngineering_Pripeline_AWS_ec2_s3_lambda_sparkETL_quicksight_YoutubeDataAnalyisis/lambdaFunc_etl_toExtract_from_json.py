import awswrangler as wr # To access the filesystem stored in S3
import pandas as pd
import urllib.parse
import os

os_input_s3_clean_data_path             = os.environ['s3_clean_data_path']      
os_input_glue_catalog_db_name           = os.environ['glue_catalog_db_name']    
os_input_glue_catalog_table_name        = os.environ['glue_catalog_table_name'] 
os_input_write_data_operation           = os.environ['write_data_operation']

def lambda_handler(event, context):
    # TODO implement
    # Get the object from the event and show its content type
    bucket  = event['Records'][0]['s3']['bucket']['name']
    key     = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    
    try:
        # Creating a DataFrame from 'items' attribute
        df_raw = wr.s3.read_json('s3://{}/{}'.format(bucket, key))
        
        # Extract required columns
        df_req = pd.json_normalize(df_raw['items'])
        
        # print(df_req)
        # Write to S3
        wr_response = wr.s3.to_parquet(
                                        df=df_req,
                                        path=os_input_s3_clean_data_path,
                                        dataset=True,
                                        database=os_input_glue_catalog_db_name,
                                        table=os_input_glue_catalog_table_name,
                                        mode=os_input_write_data_operation
                                    )
        
        return wr_response
    
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Please ensure the location is correct'.format(key, bucket))
        raise e