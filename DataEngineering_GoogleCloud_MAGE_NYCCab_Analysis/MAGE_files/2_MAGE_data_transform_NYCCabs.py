import pandas as pd
import requests
import io

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here

    # Removing Irrelevant Column
    del data['Unnamed: 0']

    # Convert Datetime from String to Datetime
    data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
    data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'])

    # Remove Duplicates and Set "trip_id" as Primary Key
    data = data.drop_duplicates().reset_index(drop=True)
    data['trip_id'] = data.index
    data.index = data['trip_id']
    
    # CREATE DIMENSION TABLES

    ## Dimension Table 1: Vendor_dim

    vendor_code_name = {
        1: "Creative Mobile Technologies, LLC",
        2: "VeriFone Inc."
    }

    Vendor_dim = data[['VendorID']].drop_duplicates().reset_index(drop=True)
    Vendor_dim = Vendor_dim.rename(columns={'VendorID': 'vendor_id'})
    Vendor_dim['vendor_name'] = Vendor_dim['vendor_id'].map(vendor_code_name)
    
    ### SET 'vendor_id' as Primary Key
    Vendor_dim.index = Vendor_dim['vendor_id']
    Vendor_dim = Vendor_dim.dropna()


    ## Dimension Table 2: Datetime_dim

    ### Pickup Dimensions
    Datetime_dim = data[['tpep_pickup_datetime', 'tpep_dropoff_datetime']].reset_index(drop=True)

    Datetime_dim['pickup_hour'] = Datetime_dim['tpep_pickup_datetime'].dt.hour
    Datetime_dim['pickup_day'] = Datetime_dim['tpep_pickup_datetime'].dt.hour
    Datetime_dim['pickup_month'] = Datetime_dim['tpep_pickup_datetime'].dt.month
    Datetime_dim['pickup_year'] = Datetime_dim['tpep_pickup_datetime'].dt.year
    Datetime_dim['pickup_weekday'] = Datetime_dim['tpep_pickup_datetime'].dt.day_name()

    ### Dropoff Dimensions    
    Datetime_dim['dropoff_hour'] = Datetime_dim['tpep_dropoff_datetime'].dt.hour
    Datetime_dim['dropoff_day'] = Datetime_dim['tpep_dropoff_datetime'].dt.hour
    Datetime_dim['dropoff_month'] = Datetime_dim['tpep_dropoff_datetime'].dt.month
    Datetime_dim['dropoff_year'] = Datetime_dim['tpep_dropoff_datetime'].dt.year
    Datetime_dim['dropoff_weekday'] = Datetime_dim['tpep_dropoff_datetime'].dt.day_name()

    ### SET 'datetime_id' as Primary Key
    Datetime_dim['datetime_id'] = Datetime_dim.index
    Datetime_dim.index = Datetime_dim['datetime_id']


    ## Dimension Table 3: PULocation_dim

    ### Loading the Taxi Zones data

    url = 'https://storage.googleapis.com/data-engg-mage-nyccabs/taxi_zones_cleaned.csv'
    response = requests.get(url)

    tlc = pd.read_csv(io.StringIO(response.text), sep=',', low_memory=False)

    PULocation_dim = tlc[['LocationID', 'zone', 'borough', 'Shape_Area', 'Shape_Leng', 'Longitude', 'Latitude']].drop_duplicates().reset_index(drop=True)
    PULocation_dim = PULocation_dim.rename(columns={'LocationID': 'pickup_location_id'})

    ### SET 'pickup_location_id' as Primary Key
    # PULocation_dim.set_index('pickup_location_id', inplace=True)
    PULocation_dim.index = PULocation_dim['pickup_location_id']
    PULocation_dim = PULocation_dim[['pickup_location_id', 'zone', 'borough', 'Shape_Area', 'Shape_Leng', 'Longitude', 'Latitude']]
    

    ## Dimension Table 4: DOLocation_dim
    DOLocation_dim = tlc[['LocationID', 'zone', 'borough', 'Shape_Area', 'Shape_Leng', 'Longitude', 'Latitude']].drop_duplicates().reset_index(drop=True)
    DOLocation_dim = DOLocation_dim.rename(columns={'LocationID': 'dropoff_location_id'})

    ### SET 'dropoff_location_id' as Primary Key
    # DOLocation_dim.set_index('dropoff_location_id', inplace=True)
    DOLocation_dim.index = DOLocation_dim['dropoff_location_id']
    DOLocation_dim = DOLocation_dim[['dropoff_location_id', 'zone', 'borough', 'Shape_Area', 'Shape_Leng', 'Longitude', 'Latitude']]

    
    
    ## Dimension Table 5: Ratecode_dim
    Rate_code_type = {
        1: "Standard rate",
        2: "JFK",
        3: "Newark",
        4: "Nassau or Westchester",
        5: "Negotiated fare",
        6: "Group ride"
    }

    Rate_code_dim = data[['RatecodeID']].drop_duplicates().reset_index(drop=True)
    Rate_code_dim = Rate_code_dim.rename(columns={'RatecodeID': 'rate_code_id'})
    Rate_code_dim['rate_code_name'] = Rate_code_dim['rate_code_id'].map(Rate_code_type)

    ### SET 'rate_code_id' as Primary Key
    # Rate_code_dim.set_index('rate_code_id', inplace=True)
    Rate_code_dim.index = Rate_code_dim['rate_code_id']
    Rate_code_dim = Rate_code_dim.dropna()  # Remove Null Values
    Rate_code_dim = Rate_code_dim[['rate_code_id', 'rate_code_name']]


    ## Dimension Table 6: Payment_dim
    Payment_type_name = {
        1: "Credit card",
        2: "Cash",
        3: "No charge",
        4: "Dispute",
        5: "Unknown",
        6: "Voided trip"
    }
    Payment_type_dim = data[['payment_type']].drop_duplicates().reset_index(drop=True)
    Payment_type_dim = Payment_type_dim.rename(columns={'payment_type': 'payment_type_id'})
    Payment_type_dim['payment_type_name'] = Payment_type_dim['payment_type_id'].map(Payment_type_name)
    # Payment_type_dim.set_index('payment_type_id', inplace=True)

    ### SET 'payment_type_id' as Primary Key
    # Payment_type_dim.rename(columns={'index': 'payment_type_id'})
    Payment_type_dim.index = Payment_type_dim['payment_type_id']
    Payment_type_dim = Payment_type_dim.dropna()
    Payment_type_dim = Payment_type_dim[['payment_type_id', 'payment_type_name']]


    # CREATE FACT TABLE (by merging the other tables)
    fact_table = data.merge(Vendor_dim, left_on='VendorID', right_index=True) \
                      .merge(Datetime_dim, left_index=True, right_index=True) \
                      .merge(PULocation_dim, left_on='PULocationID', right_index=True) \
                      .merge(DOLocation_dim, left_on='DOLocationID', right_index=True) \
                      .merge(Rate_code_dim, left_on='RatecodeID', right_index=True) \
                      .merge(Payment_type_dim, left_on='payment_type', right_index=True) \
                      [['trip_id','VendorID', 'datetime_id', 'passenger_count',
                        'trip_distance', 'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID',
                        'payment_type', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
                        'improvement_surcharge', 'total_amount']]

    # Clean and Improve the Readability of FACT Table
    fact_table = fact_table.sort_values('datetime_id')

    fact_table['trip_id'] = fact_table.reset_index().index

    fact_table.set_index('trip_id', inplace=True)

    fact_table = fact_table.dropna()

    # print(fact_table.info())

    return {"Vendor_dim":Vendor_dim.to_dict(orient="dict"),
    "Datetime_dim":Datetime_dim.to_dict(orient="dict"),
    "PULocation_dim":PULocation_dim.to_dict(orient="dict"),
    "DOLocation_dim":DOLocation_dim.to_dict(orient="dict"),
    "Rate_code_dim":Rate_code_dim.to_dict(orient="dict"),
    "Payment_type_dim":Payment_type_dim.to_dict(orient="dict"),
    "fact_table":fact_table.to_dict(orient="dict")}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
