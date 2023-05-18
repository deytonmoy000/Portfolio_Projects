-- The csv file has been downloaded from "https://www.kaggle.com/datasets/tmthyjames/nashville-housing-data?resource=download" and imported with Empty strings as NULL. The following steps are taken to clean the data after import.

------------------------------------------------
-- FORMATTING THE COLUMNS

-- Remove the whitespaces from all columns using a procedure

Delimiter //

Drop Procedure If Exists RemoveWhiteSpaceFromColumnNames //

Create Procedure
	RemoveWhiteSpaceFromColumnNames(In TblName VARCHAR(255))
Begin
	Declare done INT DEFAULT False;
	Declare oldColumnName VARCHAR(255);
	Declare newColumnName VARCHAR(255);
	Declare cur Cursor For
	 	Select column_name
	 		From information_schema.columns
	 		Where table_name = TblName;
	Declare Continue Handler For Not Found Set done = True;
	
	Open cur;
	
	read_loop: LOOP
		Fetch cur into oldColumnName;
		If done Then
			LEAVE read_loop;
		End If;
		
		Set newColumnName = Replace(oldColumnName, ' ', '');
		If oldColumnName <> newColumnName Then
		
			Set @sql = Concat('Alter Table ', TblName ,' Rename Column `', oldColumnName, '` to `', newColumnName, '`');
			SELECT @sql AS 'SQL Statement'; -- Print the SQL statement
		 	Prepare stmt From @sql;
		 	Execute stmt;
		 	Deallocate Prepare stmt;
		End If;
	End LOOP;
	
	Close cur;
End //
Delimiter ;

Call RemoveWhiteSpaceFromColumnNames('NashvilleHousing');


-- Rename the columns "Column1" and "Unnamed: 0"
ALTER TABLE
	NashvilleHousing
RENAME COLUMN
	Column1	TO S_No;
	
ALTER TABLE
	NashvilleHousing
RENAME COLUMN
	`Unnamed:0`	TO ID;

-- Update the datatype of "SaleDate" to DATE
ALTER TABLE
	NashvilleHousing
MODIFY COLUMN
	`SaleDate` date;


------------------------------------------------

-- PROPERTY ADDRESS

-- Do rows with the same "ParcelID" share the same "PropertyAddress" ?

Select 
	a.ID,
	a.ParcelID, 
	a.PropertyAddress, 
	b.ParcelID, 
	b.PropertyAddress
From
	NashvilleHousing a
Join
	NashvilleHousing b
On
	a.ParcelID = b.ParcelID
	And a.PropertyAddress <> b.PropertyAddress
Order By
	a.ParcelID;
	
-- Yes, majority of rows with the same "ParcelID" share the same "PropertyAddress", however, some of them show difference due to misplaced whitespaces.

-- Removing the extra whitespaces from the "PropertyAddress" column should fix the issue.

Update
	NashvilleHousing
Set
	PropertyAddress = Replace(PropertyAddress, '  ', ' ');
	
-- Removing the extra whitespaces from the "Address" (owner) column should help us avoid the issue for the column.

Update
	NashvilleHousing
Set
	Address = Replace(Address, '  ', ' ');


-- Are there rows with the same "ParcelID" but with one row having a NULL "PropertyAddress" while the other row is not NULL?

Select 
	a.ID,
	a.ParcelID, 
	a.PropertyAddress, 
	b.ParcelID, 
	b.PropertyAddress
From
	NashvilleHousing a
Join
	NashvilleHousing b
On
	a.ParcelID = b.ParcelID
	And a.PropertyAddress <> b.PropertyAddress
Where
	a.PropertyAddress is NULL
Order By
	a.ParcelID;


-- Yes, some rows had the "PropertyAddress" as NULL while rows with the same "ParcelID" did not.

-- Updating these rows (with NULL values) with the values of their Non-Null counterpart should fix the issue (NOTE: Since some rows still do not have a PropertyAddress; these will need to be excluded for analysis that relies on the PropertyAddress).

Update
	NashvilleHousing a
Join
	NashvilleHousing b
On
	a.ParcelID = b.ParcelID
	And a.ID <> b.ID
Set
	a.PropertyAddress = COALESCE(a.PropertyAddress, b.PropertyAddress)
Where 
	a.PropertyAddress is Null;
	

-- Are there rows with the same "OwnerName" but with one row having a NULL "Address" while the other row is not NULL?

Select 
	a.OwnerName,
	a.Address, 
	a.PropertyAddress, 
	b.Address, 
	b.PropertyAddress
From
	NashvilleHousing a
Join
	NashvilleHousing b
On
	a.OwnerName = b.OwnerName
	And a.Address <> b.Address
Where
	a.Address is NULL
Order By
	a.OwnerName;

-- No, there were no results indicating such discrepancy

-------------------------------------------------------


-- Breaking Down the PropertyAddress and Owner Address into House No. and Street for better readability of the table that will help analyze properties based on the street.

-- Creating a Function to Split the Address String based on a delimiter and Extract the specified index element

Drop Function If Exists Split_Str;

Create Function
	Split_Str(
  		x VARCHAR(255),
  		delim VARCHAR(8),
  		pos INT,
  		AllAfterPos INT
	)
Returns VARCHAR(255)
Deterministic
Begin
-- 	Set AllAfterPos = IfNull(AllAfterPos, False);
	Declare Subs VARCHAR(255);
	Set Subs = SUBSTRING_INDEX(x, delim, pos);
	
	If AllAfterPos Then
		If Subs REGEXP '^[0-9]+$' THEN
			Return 
				SUBSTRING(x, LENGTH(Subs) + 2);
		Else
			Return x;
		End If;
	End If;  
	
	If Subs REGEXP '^[0-9]+$' THEN
		Return
			Replace(
				SUBSTRING(
					Subs,
		       		LENGTH(SUBSTRING_INDEX(x, delim, pos -1)) + 1
		       	),
		       	delim, ''
		     );
	Else
		Return Null;
	End If;
End;


---------------------------------------------------

-- Adding Two Columns (House # and Street) for Splitting the Property Address

Alter Table
	NashvilleHousing
Add Column(
	PropertyHouseNo VARCHAR(255) Default Null, 
	PropertyStreet VARCHAR(255) Default Null
);

-- Updating the columns by breaking the Property Address using the Split_Str() function.

Update
	NashvilleHousing
Set
	PropertyHouseNo = Split_Str(PropertyAddress, ' ', 1, False),
	PropertyStreet = Split_str(PropertyAddress, ' ', 1, True);

	
-- Adding Two Columns (House # and Street) for Owner Address
	
Alter Table
	NashvilleHousing
Add Column(
	OwnerHouseNo VARCHAR(255) Default Null, 
	OwnerStreet VARCHAR(255) Default Null
);

-- Updating the columns by breaking the Property Address using the Split_Str() function.

Update
	NashvilleHousing
Set
	HouseNo = Split_Str(Address, ' ', 1, False),
	Street = Split_str(Address, ' ', 1, True);
	

-------------------------------------------------

-- SOLD AS VACANT

-- Check if All Values in the column "SoldAsVacant" are consistent (Yes/No).

Select 
	SoldAsVacant,
	Count(SoldAsVacant)
From
	NashvilleHousing
Group By
	SoldAsVacant;
	
-- No, there are it seems some values with 'Y' and 'N' instead of Yes/No.

-- The issue can be fixed by applying an Update using Case

Update
	NashvilleHousing
Set 
	SoldAsVacant = 	Case When SoldAsVacant = 'N' Then 'No'
					When SoldAsVacant = 'Y' Then 'Yes'
					Else SoldAsVacant
					End;
					
------------------------------------------------


-- Remove Duplicates

-- Duplicate data (rows that have identical ParcelID, PropertyAddress, SalePrice, SaleDate, OwnerName and LegalReference) are removed using Row_Number() and a cte

With cte AS (
	Select *, 
	Row_Number() Over (
		Partition By 	
				ParcelID,
				PropertyAddress,
				SaleDate,
				SalePrice,
				LegalReference,
				OwnerName
		Order By
				ID				
	) rowNum
	From
		NashvilleHousing
)
Delete From NashvilleHousing
Where
	ID in (
		Select ID from cte Where rowNum > 1
	);

-------------------------------------------------------

--  Remove Irrelevant (image, TaxDistrict), and Duplicate Columns (Eg. the S_No and ID are identical, and the previous address columns are no longer needed due to the updated segregated columns)

Alter Table
	NashvilleHousing
Drop Column S_No, 
Drop Column PropertyAddress, 
Drop Column Address, 
Drop Column image;

------------------------------------------------------