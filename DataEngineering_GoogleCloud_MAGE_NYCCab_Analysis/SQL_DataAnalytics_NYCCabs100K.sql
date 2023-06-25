-- Net and Avg Income by Vendors

SELECT
  f.VendorID, 
  v.vendor_name, 
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.total_amount) Total_Amount,
  AVG(f.total_amount) Avg_Amount_Per_Ride
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.Vendor_dim` v
ON
  f.VendorID = v.vendor_id
GROUP BY
  f.VendorID, 
  v.vendor_name
ORDER BY
  f.VendorID;


-- Total Rides and Fare/Tip by Payment Type

SELECT
  f.payment_type, 
  p.payment_type_name,
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.Payment_type_dim` p
ON
  f.payment_type = p.payment_type_id
GROUP BY
  f.payment_type, 
  p.payment_type_name
ORDER BY
  f.payment_type;


-- Net and Avg Income by Pickup Zones

SELECT
  f.PULocationID, 
  pk.zone PickupZone,
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.PULocation_dim` pk
ON
  f.PULocationID = pk.pickup_location_id
GROUP BY
  f.PULocationID, 
  pk.zone
ORDER BY
  Total_Rides DESC;


-- Top 10 Pickup Zones

SELECT
  f.PULocationID, 
  pk.zone PickupZone,
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.PULocation_dim` pk
ON
  f.PULocationID = pk.pickup_location_id
GROUP BY
  f.PULocationID, 
  pk.zone
ORDER BY
  Total_Rides DESC
LIMIT 10;


-- Net and Avg Income by Dropoff Zones

SELECT
  f.DOLocationID, 
  dl.zone DropoffZone,
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.DOLocation_dim` dl
ON
  f.DOLocationID = dl.dropoff_location_id
GROUP BY
  f.DOLocationID, 
  dl.zone
ORDER BY
  Total_Rides DESC;


-- Top 10 Dropoff Zones by Ride Count

SELECT
  f.DOLocationID, 
  dl.zone DropoffZone,
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.DOLocation_dim` dl
ON
  f.DOLocationID = dl.dropoff_location_id
GROUP BY
  f.DOLocationID, 
  dl.zone
ORDER BY
  Total_Rides DESC
LIMIT 10;

-- Top 10 Dropoff Zones by Avg Fare

SELECT
  f.DOLocationID, 
  dl.zone DropoffZone,
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.DOLocation_dim` dl
ON
  f.DOLocationID = dl.dropoff_location_id
GROUP BY
  f.DOLocationID, 
  dl.zone
ORDER BY
  Avg_Fare_Amount DESC
LIMIT 10;


-- Net and Avg Income by Pickup Borough

SELECT
  pk.borough PickupBorough,
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.PULocation_dim` pk
ON
  f.PULocationID = pk.pickup_location_id
GROUP BY
  pk.borough
ORDER BY
  Total_Rides DESC;


-- Top 10 Pickup Borough

SELECT
  pk.borough PickupBorough,
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.PULocation_dim` pk
ON
  f.PULocationID = pk.pickup_location_id
GROUP BY
  pk.borough
ORDER BY
  Total_Rides DESC
LIMIT 10;


-- Net and Avg Income by Dropoff Borough

SELECT
  dl.borough DropoffBorough,
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.DOLocation_dim` dl
ON
  f.DOLocationID = dl.dropoff_location_id
GROUP BY
  dl.borough
ORDER BY
  Total_Rides DESC;


-- Top 10 Dropoff Borough by Ride Count

SELECT
  dl.borough DropoffBorough,
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.DOLocation_dim` dl
ON
  f.DOLocationID = dl.dropoff_location_id
GROUP BY
  dl.borough
ORDER BY
  Total_Rides DESC
LIMIT 10;

-- Top 10 Dropoff Borughs by Avg Fare

SELECT
  dl.borough DropoffBorough,
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.DOLocation_dim` dl
ON
  f.DOLocationID = dl.dropoff_location_id
GROUP BY
  dl.borough
ORDER BY
  Avg_Fare_Amount DESC
LIMIT 10;


-- Top 10 Dropoff Zones by Toll Amount

SELECT
  dl.zone DropoffZone,
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.tolls_amount) Avg_Toll_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.DOLocation_dim` dl
ON
  f.DOLocationID = dl.dropoff_location_id
GROUP BY
  dl.zone
ORDER BY
  Avg_Toll_Amount DESC
LIMIT 10;

-- Top 10 Dropoff Boroughs by Toll Amount

SELECT
  dl.borough DropoffBorough,
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.tolls_amount) Avg_Toll_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.DOLocation_dim` dl
ON
  f.DOLocationID = dl.dropoff_location_id
GROUP BY
  dl.borough
ORDER BY
  Avg_Toll_Amount DESC
LIMIT 10;


-- Peak Hours (Top 10 Active Hours by Ride Count)

SELECT
  d.pickup_hour,
  COUNT (DISTINCT f.datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.Datetime_dim` d
ON
  f.datetime_id = d.datetime_id
GROUP BY
  d.pickup_hour
ORDER BY
  Total_Rides DESC
LIMIT 10;


-- Peak Hours (Top 10 Active Hours by Avg Tip Amount)

SELECT
  d.pickup_hour,
  COUNT (DISTINCT f.datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.Datetime_dim` d
ON
  f.datetime_id = d.datetime_id
GROUP BY
  d.pickup_hour
ORDER BY
  Avg_Tip_Amount DESC
LIMIT 10;


-- Activity by Day

SELECT
  d.pickup_weekday,
  COUNT (DISTINCT f.datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.Datetime_dim` d
ON
  f.datetime_id = d.datetime_id
GROUP BY
  d.pickup_weekday
ORDER BY
  Total_Rides DESC
LIMIT 10;

-- Total Rides and Income by Passenger Count

SELECT
  f.passenger_count,
  COUNT (DISTINCT f.datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
GROUP BY
  f.passenger_count
ORDER BY
  Total_Rides DESC;


-- Most Popular Routes

SELECT
  pk.zone PickupZone,
  dl.zone DropoffZone,
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.tolls_amount) Avg_Toll_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.DOLocation_dim` dl
ON
  f.DOLocationID = dl.dropoff_location_id
JOIN
  `mage-nyccab.dataengg_nyccabs100K.PULocation_dim` pk
ON
  f.PULocationID = pk.pickup_location_id
GROUP BY
  pk.zone,
  dl.zone
ORDER BY
  Total_Rides DESC
LIMIT 10;

-- Highest Toll Routes

SELECT
  pk.zone PickupZone,
  dl.zone DropoffZone,
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.tolls_amount) Avg_Toll_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.DOLocation_dim` dl
ON
  f.DOLocationID = dl.dropoff_location_id
JOIN
  `mage-nyccab.dataengg_nyccabs100K.PULocation_dim` pk
ON
  f.PULocationID = pk.pickup_location_id
GROUP BY
  pk.zone,
  dl.zone
ORDER BY
  Avg_Toll_Amount DESC
LIMIT 10;


-- Highest Tip Routes

SELECT
  pk.zone PickupZone,
  dl.zone DropoffZone,
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.tolls_amount) Avg_Toll_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.DOLocation_dim` dl
ON
  f.DOLocationID = dl.dropoff_location_id
JOIN
  `mage-nyccab.dataengg_nyccabs100K.PULocation_dim` pk
ON
  f.PULocationID = pk.pickup_location_id
GROUP BY
  pk.zone,
  dl.zone
ORDER BY
  Avg_Tip_Amount DESC
LIMIT 10;

-- Highest Avg Fare Routes

SELECT
  pk.zone PickupZone,
  dl.zone DropoffZone,
  COUNT (DISTINCT datetime_id) Total_Rides,
  SUM(f.fare_amount) Total_Fare_Amount,
  AVG(f.fare_amount) Avg_Fare_Amount,
  AVG(f.tip_amount) Avg_Tip_Amount,
  AVG(f.tolls_amount) Avg_Toll_Amount,
  AVG(f.passenger_count) Avg_Passenger_Count
FROM
  `mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.DOLocation_dim` dl
ON
  f.DOLocationID = dl.dropoff_location_id
JOIN
  `mage-nyccab.dataengg_nyccabs100K.PULocation_dim` pk
ON
  f.PULocationID = pk.pickup_location_id
GROUP BY
  pk.zone,
  dl.zone
ORDER BY
  Avg_Fare_Amount DESC
LIMIT 10;