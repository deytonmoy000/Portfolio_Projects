# Data Engineering Pipeline using Apache Kafka (Producer, Consumer) and AWS (EC2, S3) for Real-Time Database Update (Crawler, Athena)

## Tasks Performed:

- Instantiate/ Run a AWS Cluster (**EC2**)

- Start **Kafka Zookeeper**

- Start **Kafka Server**

- Create a **Topic**

- Fetch the data using a stock market API (https://www.alphavantage.co/support/#api-key%22)

- Send the Data using **Kafka Producer** (*Kafka_Producer_StreamingStockMarket.ipynb*)

- Receive the Data using Kafka Consumer and Store the Data in **AWS S3** (*Kafka_Consumer_StreamingStockMarket.ipynb*)

- Run the **AWS Glue Crawler** to automate database update

- Use the **AWS Athena** for data analysis.