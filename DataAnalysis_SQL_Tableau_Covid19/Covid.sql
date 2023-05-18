-- The csv file has been downloaded from "https://covid.ourworldindata.org/data/owid-covid-data.csv" and imported with Empty strings as NULL. The following steps are taken to analyze and visualize the trends in the data after import.

-- Select Data that we are going to be using for Covid Deaths
SELECT
	location,
	DATE,
	total_cases,
	new_cases,
	total_deaths,
	population
FROM
	Covid
ORDER BY
	location,
	DATE;


-- Total Cases vs Deaths

SELECT
	location,
	DATE,
	total_cases,
	total_deaths,
	(total_deaths / total_cases)* 100 as death_percentage
FROM
	Covid
ORDER BY
	location,
	DATE;
	
-- Peak Death % by Country

SELECT
	location,
	max((total_deaths/total_cases)*100) as max_death_percentage
FROM
	Covid
WHERE 
	continent is not null
group by
	location
having 
	max_death_percentage < 3
ORDER BY
	location;


-- Total Cases vs Population

SELECT
	location,
	DATE,
	population,
	total_cases,
	(total_cases / population)* 100 as infection_rate
FROM
	Covid
WHERE 
	continent is not null
ORDER BY
	location,
	DATE;


--  Max Population % Infected by Country

SELECT
	location,
	population,
	max(total_cases) highest_infection_count,
	max(new_cases) highest_one_day_infection_count,
	max(total_cases / population)* 100 as percentage_population_infected
FROM
	Covid
WHERE 
	continent is not null
GROUP BY
	location,
	population
ORDER BY
	highest_one_day_infection_count desc;


-- Countries wit Highest Death Count Per Population by Country

SELECT
	location,
	population,
	max(total_deaths) highest_death_count,
	max(new_deaths) highest_one_day_death_count,
	max(total_deaths / population)* 100 as percentage_population_death
FROM
	Covid
WHERE 
	continent is not null
GROUP BY
	location,
	population
ORDER BY
	percentage_population_death desc;

-- Countries wit Highest Death Count Per Population by Continent

SELECT
	location,
	population,
	max(total_deaths) highest_death_count
FROM
	Covid
WHERE 
	continent is null and
	location not like "%ncome%"
GROUP BY
	location, population
ORDER BY
	highest_death_count desc;
	
	
-- Summarizing the death percentage based on Income Group

SELECT
	location,
	population,
	max(total_deaths),
	max(total_deaths/population)*100 highest_death_percentage
FROM
	Covid
WHERE 
	continent is null
GROUP BY
	location, population
HAVING
	location like "%ncome%"
ORDER BY
	highest_death_percentage desc	;
	

-- Global Numbers By Day

SELECT
	
	DATE,
	SUM(population) PopulationWorldwide,
	SUM(new_tests) NewTestsWorldwide,
	SUM(new_cases) NewCasesWorldwide,
	SUM(new_deaths) NewDeathsWorldwide,
	SUM(total_tests) TotalTestsWorldwide, 
	SUM(total_cases) TotalCasesWorldwide,
	SUM(total_deaths) TotalDeathsWorldwide, 
	SUM(total_deaths)/SUM(total_cases)*100 DeathPercentageWorldwide, 
	SUM(new_vaccinations) NewVaccinationsWorldwide,
	SUM(total_vaccinations) TotalVaccinationsWorldwide,
	SUM(total_boosters) TotalBoostersWorldwide,
	SUM(people_vaccinated) TotalPeopleVaccinated,
	SUM(people_fully_vaccinated) TotalPeopleFullyVaccinated,
	AVG(median_age) MedianAgeWorldwide,
	SUM(icu_patients) ICUPatientsWorldwide,
	SUM(hosp_patients) HospitalizedWorldwide
FROM
	Covid
WHERE 
	continent is not null
GROUP BY
	DATE
ORDER BY
	DATE;

-- Global Numbers By Income Group and Day

SELECT
	location,
	DATE,
	population,
	SUM(new_tests) NewTestsWorldwide,
	SUM(new_cases) NewCasesWorldwide,
	SUM(new_deaths) NewDeathsWorldwide,
	SUM(total_tests) TotalTestsWorldwide, 
	SUM(total_cases) TotalCasesWorldwide,
	SUM(total_deaths) TotalDeathsWorldwide, 
	SUM(total_deaths)/SUM(total_cases)*100 DeathPercentageWorldwide, 
	SUM(new_vaccinations) NewVaccinationsWorldwide,
	SUM(total_vaccinations) TotalVaccinationsWorldwide,
	SUM(total_boosters) TotalBoostersWorldwide,
	SUM(people_vaccinated) TotalPeopleVaccinated,
	SUM(people_fully_vaccinated) TotalPeopleFullyVaccinated,
	AVG(median_age) MedianAgeWorldwide,
	SUM(icu_patients) ICUPatientsWorldwide,
	SUM(hosp_patients) HospitalizedWorldwide
FROM
	Covid
WHERE 
	continent is null
	and location like "%ncome%"
GROUP BY
	DATE, location, population
ORDER BY
	DATE;

-- Global Numbers By Income Group 
SELECT
	location,
	population,
	SUM(new_tests) TotalTestsWorldwide,
	SUM(new_cases) TotalCasesWorldwide,
	SUM(new_deaths) TotalDeathsWorldwide,
	SUM(new_deaths)/SUM(new_cases)*100 DeathPercentageWorldwide, 
	SUM(new_vaccinations) TotalVaccinationsWorldwide,
	MAX(total_boosters) TotalBoostersWorldwide,
	MAX(people_vaccinated) TotalPeopleVaccinated,
	MAX(people_fully_vaccinated) TotalPeopleFullyVaccinated,
	AVG(median_age) MedianAgeWorldwide,
	AVG(icu_patients) AvgICUPatientsWorldwide,
	AVG(hosp_patients) AvgHospitalizedWorldwide
FROM
	Covid
WHERE 
	continent is null
	and location like "%ncome%"
GROUP BY
	location, population
ORDER BY
	location;



-- Global Numbers By Country

SELECT
	location,
	population,
	SUM(new_tests) TotalTests,
	SUM(new_cases) TotalCases,
	SUM(new_deaths) TotalDeaths,
	SUM(new_deaths)/SUM(new_cases)*100 DeathPercentage, 
	SUM(new_vaccinations) TotalVaccinations,
	MAX(total_boosters) TotalBoosters,
	MAX(people_vaccinated) TotalPeopleVaccinated,
	MAX(people_fully_vaccinated) TotalPeopleFullyVaccinated,
	AVG(median_age) MedianAgeWorldwide,
	AVG(icu_patients) AvgICUPatientsWorldwide,
	AVG(hosp_patients) AvgHospitalizedWorldwide
FROM
	Covid
WHERE 
	continent is not null
GROUP BY
	location, population
ORDER BY
	location;
	
-- Global Numbers By Region

SELECT
	location,
	DATE,
	population,
	SUM(new_tests) NewTests,
	SUM(new_cases) NewCases,
	SUM(new_deaths) NewDeaths,
	SUM(new_deaths)/SUM(new_cases)*100 DeathPercentage, 
	SUM(new_vaccinations) NewVaccinations,
	MAX(total_boosters) TotalBoosters,
	MAX(people_vaccinated) TotalPeopleVaccinated,
	MAX(people_fully_vaccinated) TotalPeopleFullyVaccinated,
	AVG(median_age) MedianAgeWorldwide,
	AVG(icu_patients) AvgICUPatientsWorldwide,
	AVG(hosp_patients) AvgHospitalizedWorldwide
FROM
	Covid
WHERE 
	continent is null
	AND location not like "%ncome%"
	AND location not like "World"
	AND location not like "%Union"
GROUP BY
	DATE, location, population
ORDER BY
	DATE;


-- Global Numbers By Continent and Date (using Parition By)

SELECT
	location,
	DATE,
	population,
	SUM(new_cases) OVER (PARTITION BY location order by location, date) TotalCases,
	SUM(new_deaths) OVER (PARTITION BY location order by location, date) TotalDeaths,
	SUM(new_vaccinations) OVER (PARTITION BY location order by location, date) TotalVaccinations,
	SUM(new_cases) OVER (PARTITION BY location order by location, date) TotalCases	
FROM
	Covid
WHERE 
	continent is null
	AND location not like "%ncome%"
	AND location not like "World"
	AND location not like "%Union"
ORDER BY
	location, date;


-- Global Numbers By Country and Date

SELECT
	location,
	continent,
	DATE,
	population,
	new_cases,
	total_cases,
	new_deaths,
	total_deaths,
	new_vaccinations,
	total_vaccinations,
	people_vaccinated,
	people_fully_vaccinated,
	total_boosters,
	hosp_patients,
	icu_patients
FROM
	Covid
WHERE 
	continent is not null
ORDER BY
	location,
	DATE;

-- Global Numbers By Continent and Day

SELECT
	continent,
	DATE,
	SUM(new_tests) TotalTests,
	SUM(new_cases) TotalCases,
	SUM(new_deaths) TotalDeaths,
	SUM(new_deaths)/SUM(new_cases)*100 DeathPercentage, 
	SUM(new_vaccinations) TotalVaccinations,
	MAX(total_boosters) TotalBoosters,
	MAX(people_vaccinated) TotalPeopleVaccinated,
	MAX(people_fully_vaccinated) TotalPeopleFullyVaccinated,
	AVG(median_age) MedianAgeWorldwide,
	AVG(icu_patients) AvgICUPatientsWorldwide,
	AVG(hosp_patients) AvgHospitalizedWorldwide
FROM
	Covid
WHERE 
	continent is not null
Group By
	continent, date
ORDER BY
	continent,
	DATE;

-- Global Numbers By Continent (using Group By continent; Will be used for verification with the prior result on Global Numbers By Continent using 'location' column)

SELECT
	continent,
	SUM(new_tests) TotalTests,
	SUM(new_cases) TotalCases,
	SUM(new_deaths) TotalDeaths,
	SUM(new_deaths)/SUM(new_cases)*100 DeathPercentage, 
	SUM(new_vaccinations) TotalVaccinations,
	MAX(total_boosters) TotalBoosters,
	MAX(people_vaccinated) TotalPeopleVaccinated,
	MAX(people_fully_vaccinated) TotalPeopleFullyVaccinated,
	AVG(median_age) MedianAgeWorldwide,
	AVG(icu_patients) AvgICUPatientsWorldwide,
	AVG(hosp_patients) AvgHospitalizedWorldwide
FROM
	Covid
WHERE 
	continent is not null
Group By
	continent
ORDER BY
	continent;



-- Using CTE

-- Continent: Population Vs. Death and Vaccination

With VsPop_Continent as
(SELECT
	location,
	DATE,
	population,
	new_cases,
	SUM(new_cases) OVER (PARTITION BY location order by location, date) TotalCases,
	new_deaths,
	SUM(new_deaths) OVER (PARTITION BY location order by location, date) TotalDeaths,
	new_vaccinations,
	SUM(new_vaccinations) OVER (PARTITION BY location order by location, date) TotalVaccinations,
	people_vaccinated,
	people_fully_vaccinated
FROM
	Covid
WHERE 
	continent is null
	AND location not like "%ncome%"
	AND location not like "World"
	AND location not like "%Union"
ORDER BY
	location, date);

-- Generate a result by Day	
-- Select 
-- 	*, 
-- 	(TotalDeaths/TotalCases)*100 as Cases_death_percent,
-- 	(TotalDeaths/population)*100 as Population_death_percent,
-- 	(people_vaccinated/population)*100 as Population_vaccinated_percent
-- From 
-- 	VsPop_Continent


-- Generate a Overall Summary

Select 
	location,
	AVG(population) Population,
	MAX(TotalCases) Cases,
	MAX(TotalDeaths) Deaths,
	MAX(people_vaccinated) Vaccinated,
	(SUM(TotalDeaths)/SUM(TotalCases))*100 as Cases_death_percent,
	(SUM(TotalDeaths)/SUM(population))*100 as Population_death_percent,
	(MAX(people_vaccinated)/AVG(population))*100 as Population_vaccinated_percent
From 
	VsPop_Continent
Group by
	location;


-- Country: Population Vs. Death and Vaccination

With VsPop_Country as
(SELECT
	continent,
	location,
	DATE,
	population,
	new_cases,
	SUM(new_cases) OVER (PARTITION BY location order by location, date) TotalCases,
	new_deaths,
	SUM(new_deaths) OVER (PARTITION BY location order by location, date) TotalDeaths,
	new_vaccinations,
	SUM(new_vaccinations) OVER (PARTITION BY location order by location, date) TotalVaccinations,
	people_vaccinated,
	people_fully_vaccinated
FROM
	Covid
WHERE 
	continent is not null
ORDER BY
	location, date);


-- Daily Report
-- Select 
-- 	*, 
-- 	(TotalDeaths/TotalCases)*100 as Cases_death_percent,
-- 	(TotalDeaths/population)*100 as Population_death_percent,
-- 	(people_vaccinated/population)*100 as Population_vaccinated_percent
-- From 
-- 	VsPop_Country

	
-- Generate a Overall Summary by Country

Select
	continent,
	location,
	AVG(population) Population,
	MAX(TotalCases) Cases,
	MAX(TotalDeaths) Deaths,
	MAX(people_vaccinated) Vaccinated,
	(MAX(TotalDeaths)/MAX(TotalCases))*100 as Cases_death_percent,
	(MAX(TotalDeaths)/MAX(population))*100 as Population_death_percent,
	(MAX(people_vaccinated)/MAX(population))*100 as Population_vaccinated_percent
From 
	VsPop_Country
Group by
	continent,location;



-- TEMP Table
 

Drop Table if Exists tmp_PopVs_Country;

Create TEMPORARY Table tmp_PopVs_Country (Continent nvarchar(255), Country nvarchar(255), Population numeric, Cases numeric, Deaths numeric, Vaccinated numeric, people_vaccinated numeric, people_fully_vaccinated numeric);

INSERT INTO
	tmp_PopVs_Country	
SELECT
	continent,
	location,
	population,
	SUM(new_cases) TotalCases,
	SUM(new_deaths) TotalDeaths,
	MAX(total_vaccinations) Vaccinated,
	MAX(people_vaccinated),
	MAX(people_fully_vaccinated)
FROM
	Covid
WHERE 
	continent is not null
Group By
	continent, location, population
ORDER BY
	continent, location, population;

Select * from tmp_PopVs_Country;


-- Generating a Continental Summary Using Every COuntry Data (for prior result verification)
Select 
	 continent,
	 SUM(Population) Population,
	 SUM(Cases) Cases,
	 SUM(Deaths) Deaths,
	 SUM(Vaccinated) Vaccinated,
	 (SUM(Deaths)/SUM(Cases))*100 as Cases_death_percent,
	(SUM(Deaths)/SUM(Population))*100 as Population_death_percent,
	(SUM(Vaccinated)/SUM(Population))*100 as Population_vaccinated_percent
From 
	tmp_PopVs_Country
Group by
	continent ;


-- The following query was used to confirm a discrepancy between the sum of 'new_vaccinations' ovet the entire duration and the 'total_vaccination' for every country

SELECT 
	location,
	SUM(new_vaccinations),
	MAX(total_vaccinations),
	MAX(people_vaccinated), 
	MAX(people_fully_vaccinated)
FROM
	Covid
GROUP BY
	location;

-- Gathering the Data using Views


--  View to obtain Country Wise Data Summary

DROP VIEW IF EXISTS PopVs_Country;

CREATE VIEW PopVs_Country AS
SELECT
	continent,
	location,
	population,
-- 	SUM(new_cases) TotalCases_agg,
	MAX(total_cases) TotalCases,
	MAX(total_cases_per_million) TotalCases_PerMillion,
-- 	SUM(new_deaths) TotalDeaths_agg,
	MAX(total_deaths) TotalDeaths,
	(MAX(total_deaths)/MAX(total_cases))*100 as Cases_Death_Percent,
	(MAX(total_deaths)/population)*100 as Population_Death_Percent,
-- 	SUM(new_vaccinations) Vaccinations_agg,
	MAX(total_vaccinations) Vaccinations,
	MAX(people_vaccinated) Vaccinated,
	MAX(people_fully_vaccinated) Fully_Vaccinated,
	MAX(people_fully_vaccinated_per_hundred) FullyVaccinated_PerHundred,
	MAX(total_boosters_per_hundred) TotalBoosters_PerHundred,
	(MAX(people_vaccinated)/population)*100 as Population_Vaccinated_Percent,
	(MAX(people_fully_vaccinated)/population)*100 as Population_FullyVaccinated_Percent
	
FROM
	Covid
WHERE 
	continent is not null
Group By
	location, population, continent 
ORDER BY
	continent, location, population;


--  View to obtain Continent Wise Data Summary

DROP VIEW IF EXISTS PopVs_Continent;

CREATE VIEW PopVs_Continent AS
SELECT
	continent,
	SUM(population) Population,
	SUM(TotalCases) TotalCases,
-- 	SUM(new_deaths) TotalDeaths_agg,
	SUM(TotalDeaths) TotalDeaths,
	(SUM(TotalDeaths)/SUM(TotalCases))*100 as Cases_Death_Percent,
	(SUM(TotalDeaths)/SUM(population))*100 as Population_Death_Percent,
-- 	SUM(new_vaccinations) Vaccinations_agg,
	SUM(Vaccinations) Vaccinations,
	SUM(Vaccinated) Vaccinated,
	SUM(Fully_Vaccinated) Fully_Vaccinated,
	(SUM(Vaccinated)/SUM(population))*100 as Population_Vaccinated_Percent
	
FROM
	PopVs_Country
WHERE 
	continent is not null
Group By
	continent 
ORDER BY
	continent, population;


--  View to obtain Country Wise Daily Data Summary

DROP VIEW IF EXISTS PopVs_Country_Daily;

CREATE VIEW PopVs_Country_Daily AS
SELECT
	continent,
	location,
	date,
	population,
	new_cases,
	total_cases,
	SUM(new_cases) OVER (PARTITION BY location order by location, date) total_cases_agg,
	new_deaths,
	total_deaths,
	SUM(new_deaths) OVER (PARTITION BY location order by location, date) total_deaths_agg,
	(total_deaths/total_cases)*100 as Cases_Death_Percent,
	(total_deaths/population)*100 as Population_Death_Percent,
	new_vaccinations,
	total_vaccinations,
	SUM(new_vaccinations) OVER (PARTITION BY location order by location, date) total_vaccinations_agg,
	people_vaccinated,
	people_fully_vaccinated,
	(people_vaccinated/population)*100 as Population_Vaccinated_Percent
FROM
	Covid
WHERE 
	continent is not null
-- Group By
-- 	location, continent, date, population 
ORDER BY
	location, continent, population, date;
	
	
--  View to obtain Continent Wise Daily Data Summary

DROP VIEW IF EXISTS PopVs_Continent_Daily;

CREATE VIEW PopVs_Continent_Daily AS
SELECT
	continent,
	date,
	SUM(population) Population,
	SUM(new_cases) New_Cases,
	SUM(total_cases) Total_Cases,
	SUM(total_cases_agg) Total_Cases_agg,
	SUM(new_deaths) New_Deaths,
	SUM(total_deaths) Total_Deaths,
	SUM(total_deaths_agg) Total_Deaths_agg,
	(SUM(total_deaths)/SUM(total_cases))*100 as Cases_Death_Percent,
	(SUM(total_deaths)/SUM(population))*100 as Population_Death_Percent,
	SUM(new_vaccinations) New_Vaccinations,
	SUM(total_vaccinations) Total_Vaccinations,
	SUM(total_vaccinations_agg) Total_Vaccinations_agg,
	SUM(people_vaccinated) Vaccinated,
	SUM(people_fully_vaccinated) Fully_Vaccinated,
	(SUM(people_vaccinated)/SUM(population))*100 as Population_Vaccinated_Percent
FROM
	PopVs_Country_Daily
WHERE 
	continent is not null
Group By
	continent, date 
ORDER BY
	continent, date;


--  View to obtain Region Wise (including World) Daily Data Summary For Verification (using continent is Null clause)

DROP VIEW IF EXISTS PopVs_Continent_Daily_wWorld;

CREATE VIEW PopVs_Continent_Daily_wWorld AS
SELECT
	location,
	date,
	population,
	new_cases,
	total_cases,
	SUM(new_cases) OVER (PARTITION BY location order by location, date) total_cases_agg,
	new_deaths,
	total_deaths,
	SUM(new_deaths) OVER (PARTITION BY location order by location, date) total_deaths_agg,
	(total_deaths/total_cases)*100 as Cases_Death_Percent,
	(total_deaths/population)*100 as Population_Death_Percent,
	new_vaccinations,
	total_vaccinations,
	SUM(new_vaccinations) OVER (PARTITION BY location order by location, date) total_vaccinations_agg,
	people_vaccinated,
	people_fully_vaccinated,
	(people_vaccinated/population)*100 as Population_Vaccinated_Percent
FROM
	Covid
WHERE 
	continent is null
	AND location not like "%ncome%"
	AND location not like "%Union"
-- Group By
-- 	continent, date 
ORDER BY
	location, date;


--  View to obtain Region Wise (including World) Data Summary For Verification (using continent is Null clause)

DROP VIEW IF EXISTS PopVs_Continent_wWorld;

CREATE VIEW PopVs_Continent_wWorld AS
SELECT
	location,
	population,
-- 	SUM(new_cases) TotalCases_agg,
	MAX(total_cases) TotalCases,
-- 	SUM(new_deaths) TotalDeaths_agg,
	MAX(total_deaths) TotalDeaths,
	(MAX(total_deaths)/MAX(total_cases))*100 as Cases_Death_Percent,
	(MAX(total_deaths)/population)*100 as Population_Death_Percent,
-- 	SUM(new_vaccinations) Vaccinations_agg,
	MAX(total_vaccinations) Vaccinations,
	MAX(people_vaccinated) Vaccinated,
	MAX(people_fully_vaccinated) Fully_Vaccinated,
	(MAX(people_vaccinated)/population)*100 as Population_Vaccinated_Percent
	
FROM
	Covid

WHERE 
	continent is null
	AND location not like "%ncome%"
	AND location not like "%Union"
Group By
	location, population
ORDER BY
	location;