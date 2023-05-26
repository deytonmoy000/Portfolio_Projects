import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Script generated for node Statistics
Statistics_node1684970853533 = glueContext.create_dynamic_frame.from_catalog(
    database="dataengg-youtube-clean-db-tonmoy",
    table_name="clean_statistics",
    transformation_ctx="Statistics_node1684970853533",
)

# Script generated for node Reference_Data
Reference_Data_node1684970842633 = glueContext.create_dynamic_frame.from_catalog(
    database="dataengg-youtube-clean-db-tonmoy",
    table_name="clean_statistics_reference_data",
    transformation_ctx="Reference_Data_node1684970842633",
)

# Script generated for node Join
Join_node1684970877432 = Join.apply(
    frame1=Statistics_node1684970853533,
    frame2=Reference_Data_node1684970842633,
    keys1=["category_id"],
    keys2=["id"],
    transformation_ctx="Join_node1684970877432",
)

# Script generated for node Amazon S3
AmazonS3_node1684971586599 = glueContext.getSink(
    path="s3://dataengg-youtube-analytics-useast1-dev-tonmoy",
    connection_type="s3",
    updateBehavior="UPDATE_IN_DATABASE",
    partitionKeys=["region", "category_id"],
    compression="snappy",
    enableUpdateCatalog=True,
    transformation_ctx="AmazonS3_node1684971586599",
)
AmazonS3_node1684971586599.setCatalogInfo(
    catalogDatabase="db_dataengg_youtube_analytics_tonmoy",
    catalogTableName="final_analytics_data",
)
AmazonS3_node1684971586599.setFormat("glueparquet")
AmazonS3_node1684971586599.writeFrame(Join_node1684970877432)
job.commit()
