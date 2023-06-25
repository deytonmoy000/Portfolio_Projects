CREATE OR REPLACE TABLE `mage-nyccab.dataengg_nyccabs100K.tbl_analytics` AS (
SELECT 
	f.datetime_id as trip_id,
	f.VendorID,
	v.vendor_name,
	d.tpep_pickup_datetime,
  	d.pickup_hour,
  	d.pickup_weekday,
  	d.pickup_month,
	d.tpep_dropoff_datetime,
  	d.dropoff_hour,
  	d.dropoff_weekday,
  	d.dropoff_month,
	f.passenger_count,
	f.trip_distance,
	r.rate_code_name,
	f.PULocationID,
	pk.zone PickupZone,
	pk.borough PickupBorough,
	pk.Latitude PickupLatitude,
	pk.Longitude PickupLongitude,
	f.DOLocationID,
	dl.zone DropoffZone,
	dl.borough DropoffBorough,
	dl.Latitude DropoffLatitude,
	dl.Longitude DropoffLongitude,
	p.payment_type_name,
	f.fare_amount,
	f.extra,
	f.mta_tax,
	f.tip_amount,
	f.tolls_amount,
	f.improvement_surcharge,
	f.total_amount
FROM 

`mage-nyccab.dataengg_nyccabs100K.fact_table` f
JOIN
  `mage-nyccab.dataengg_nyccabs100K.Vendor_dim` v
ON
  f.VendorID = v.vendor_id
JOIN 
	`mage-nyccab.dataengg_nyccabs100K.Datetime_dim` d  
	ON 
	f.datetime_id=d.datetime_id
JOIN 
	`mage-nyccab.dataengg_nyccabs100K.Rate_code_dim` r 
	ON 
	r.rate_code_id=f.RatecodeID  
JOIN 
	`mage-nyccab.dataengg_nyccabs100K.PULocation_dim` pk 
	ON 
	pk.pickup_location_id=f.PULocationID
JOIN 
	`mage-nyccab.dataengg_nyccabs100K.DOLocation_dim` dl 
	ON 
	dl.dropoff_location_id=f.DOLocationID
JOIN 
	`mage-nyccab.dataengg_nyccabs100K.Payment_type_dim` p 
	ON 
	p.payment_type_id=f.payment_type);