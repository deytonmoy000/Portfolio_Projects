# Data Cleaning of the Nashville Housing Data

*Data Source: ![Kaggle](https://www.kaggle.com/datasets/tmthyjames/nashville-housing-data?resource=download)

## Tasks Performed:

- Formatting the Columns: Removed whitespace (if any) from every column using **stored procedure**

- Rename columns for better readability.

- Update column datatype to avoid inaccuracy in analysis.

- Remove additional whitespaces from column values to improve consistency.

- Update column values (*PropertyAddress*) using matching information from secondary column data (*ParcelID*).

- Update column values (*OwnerAddress*) using matching information from secondary column data (*OwnerID*).

- Creating a **stored function** *Split_Str* to split string based on a delimiter and extract the specified index element.

- Breaking Down the *PropertyAddress* and *OwnerAddress* into *House No*. and *Street* using the ***Split_Str*** function for better readability of the table that will help analyze properties based on the street.

- Check the consistency of the column *SoldAsVacant* (Yes/Y, No/N) and update all the values to be consistent (Yes/No)

- Remove duplicates using the *Row_Number()* function.

- Remove irrelevant (eg. *image*) and duplicate columns (eg. the *S_No* and *ID* are identical, and the concatenated address columns are no longer needed due to the updated segregated columns). 