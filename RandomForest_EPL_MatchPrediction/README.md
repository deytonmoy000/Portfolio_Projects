# ENGLISH PREMIER LEAGUE MATCH PREDICTION USING RANDOM FOREST

### DATA SCRAPING


```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
```


```python
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
```


```python
data = requests.get(standings_url)
data
```




    <Response [200]>




```python
soup = BeautifulSoup(data.text)

standings_table = soup.select('table.stats_table')[0]

links = standings_table.find_all('a')

links = [l.get("href") for l in links]

links = [l for l in links if '/squads/' in l]
```


```python
team_urls = [f"https://fbref.com{l}" for l in links]
```


```python
data = requests.get(team_urls[0])
```


```python
matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
matches.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>Comp</th>
      <th>Round</th>
      <th>Day</th>
      <th>Venue</th>
      <th>Result</th>
      <th>GF</th>
      <th>GA</th>
      <th>Opponent</th>
      <th>xG</th>
      <th>xGA</th>
      <th>Poss</th>
      <th>Attendance</th>
      <th>Captain</th>
      <th>Formation</th>
      <th>Referee</th>
      <th>Match Report</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-07-30</td>
      <td>17:00</td>
      <td>Community Shield</td>
      <td>FA Community Shield</td>
      <td>Sat</td>
      <td>Neutral</td>
      <td>L</td>
      <td>1</td>
      <td>3</td>
      <td>Liverpool</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>57</td>
      <td>NaN</td>
      <td>Rúben Dias</td>
      <td>4-3-3</td>
      <td>Craig Pawson</td>
      <td>Match Report</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-08-07</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 1</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>2</td>
      <td>0</td>
      <td>West Ham</td>
      <td>2.2</td>
      <td>0.5</td>
      <td>75</td>
      <td>62443.0</td>
      <td>İlkay Gündoğan</td>
      <td>4-3-3</td>
      <td>Michael Oliver</td>
      <td>Match Report</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-08-13</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 2</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>4</td>
      <td>0</td>
      <td>Bournemouth</td>
      <td>1.7</td>
      <td>0.1</td>
      <td>67</td>
      <td>53453.0</td>
      <td>İlkay Gündoğan</td>
      <td>4-2-3-1</td>
      <td>David Coote</td>
      <td>Match Report</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-08-21</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 3</td>
      <td>Sun</td>
      <td>Away</td>
      <td>D</td>
      <td>3</td>
      <td>3</td>
      <td>Newcastle Utd</td>
      <td>2.1</td>
      <td>1.8</td>
      <td>69</td>
      <td>52258.0</td>
      <td>İlkay Gündoğan</td>
      <td>4-3-3</td>
      <td>Jarred Gillett</td>
      <td>Match Report</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-08-27</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 4</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>4</td>
      <td>2</td>
      <td>Crystal Palace</td>
      <td>2.2</td>
      <td>0.1</td>
      <td>74</td>
      <td>53112.0</td>
      <td>Kevin De Bruyne</td>
      <td>4-2-3-1</td>
      <td>Darren England</td>
      <td>Match Report</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
soup = BeautifulSoup(data.text)
links = soup.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if l and 'all_comps/shooting/' in l]
```


```python
data = requests.get(f"https://fbref.com{links[0]}")
```


```python
shooting = pd.read_html(data.text, match="Shooting")[0]
shooting.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">For Manchester City</th>
      <th>...</th>
      <th colspan="4" halign="left">Standard</th>
      <th colspan="5" halign="left">Expected</th>
      <th>Unnamed: 25_level_0</th>
    </tr>
    <tr>
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>Comp</th>
      <th>Round</th>
      <th>Day</th>
      <th>Venue</th>
      <th>Result</th>
      <th>GF</th>
      <th>GA</th>
      <th>Opponent</th>
      <th>...</th>
      <th>Dist</th>
      <th>FK</th>
      <th>PK</th>
      <th>PKatt</th>
      <th>xG</th>
      <th>npxG</th>
      <th>npxG/Sh</th>
      <th>G-xG</th>
      <th>np:G-xG</th>
      <th>Match Report</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-07-30</td>
      <td>17:00</td>
      <td>Community Shield</td>
      <td>FA Community Shield</td>
      <td>Sat</td>
      <td>Neutral</td>
      <td>L</td>
      <td>1</td>
      <td>3</td>
      <td>Liverpool</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Match Report</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-08-07</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 1</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>2</td>
      <td>0</td>
      <td>West Ham</td>
      <td>...</td>
      <td>18.7</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>2.2</td>
      <td>1.4</td>
      <td>0.11</td>
      <td>-0.2</td>
      <td>-0.4</td>
      <td>Match Report</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-08-13</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 2</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>4</td>
      <td>0</td>
      <td>Bournemouth</td>
      <td>...</td>
      <td>17.5</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.7</td>
      <td>1.7</td>
      <td>0.09</td>
      <td>1.3</td>
      <td>1.3</td>
      <td>Match Report</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-08-21</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 3</td>
      <td>Sun</td>
      <td>Away</td>
      <td>D</td>
      <td>3</td>
      <td>3</td>
      <td>Newcastle Utd</td>
      <td>...</td>
      <td>16.2</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.1</td>
      <td>2.1</td>
      <td>0.10</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>Match Report</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-08-27</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 4</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>4</td>
      <td>2</td>
      <td>Crystal Palace</td>
      <td>...</td>
      <td>14.1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.2</td>
      <td>2.2</td>
      <td>0.13</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>Match Report</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
shooting.columns = shooting.columns.droplevel()
shooting.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>Comp</th>
      <th>Round</th>
      <th>Day</th>
      <th>Venue</th>
      <th>Result</th>
      <th>GF</th>
      <th>GA</th>
      <th>Opponent</th>
      <th>...</th>
      <th>Dist</th>
      <th>FK</th>
      <th>PK</th>
      <th>PKatt</th>
      <th>xG</th>
      <th>npxG</th>
      <th>npxG/Sh</th>
      <th>G-xG</th>
      <th>np:G-xG</th>
      <th>Match Report</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-07-30</td>
      <td>17:00</td>
      <td>Community Shield</td>
      <td>FA Community Shield</td>
      <td>Sat</td>
      <td>Neutral</td>
      <td>L</td>
      <td>1</td>
      <td>3</td>
      <td>Liverpool</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Match Report</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-08-07</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 1</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>2</td>
      <td>0</td>
      <td>West Ham</td>
      <td>...</td>
      <td>18.7</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>2.2</td>
      <td>1.4</td>
      <td>0.11</td>
      <td>-0.2</td>
      <td>-0.4</td>
      <td>Match Report</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-08-13</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 2</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>4</td>
      <td>0</td>
      <td>Bournemouth</td>
      <td>...</td>
      <td>17.5</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.7</td>
      <td>1.7</td>
      <td>0.09</td>
      <td>1.3</td>
      <td>1.3</td>
      <td>Match Report</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-08-21</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 3</td>
      <td>Sun</td>
      <td>Away</td>
      <td>D</td>
      <td>3</td>
      <td>3</td>
      <td>Newcastle Utd</td>
      <td>...</td>
      <td>16.2</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.1</td>
      <td>2.1</td>
      <td>0.10</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>Match Report</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-08-27</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 4</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>4</td>
      <td>2</td>
      <td>Crystal Palace</td>
      <td>...</td>
      <td>14.1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.2</td>
      <td>2.2</td>
      <td>0.13</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>Match Report</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
team_data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>Comp</th>
      <th>Round</th>
      <th>Day</th>
      <th>Venue</th>
      <th>Result</th>
      <th>GF</th>
      <th>GA</th>
      <th>Opponent</th>
      <th>...</th>
      <th>Formation</th>
      <th>Referee</th>
      <th>Match Report</th>
      <th>Notes</th>
      <th>Sh</th>
      <th>SoT</th>
      <th>Dist</th>
      <th>FK</th>
      <th>PK</th>
      <th>PKatt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-07-30</td>
      <td>17:00</td>
      <td>Community Shield</td>
      <td>FA Community Shield</td>
      <td>Sat</td>
      <td>Neutral</td>
      <td>L</td>
      <td>1</td>
      <td>3</td>
      <td>Liverpool</td>
      <td>...</td>
      <td>4-3-3</td>
      <td>Craig Pawson</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>14</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-08-07</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 1</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>2</td>
      <td>0</td>
      <td>West Ham</td>
      <td>...</td>
      <td>4-3-3</td>
      <td>Michael Oliver</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>13</td>
      <td>1</td>
      <td>18.7</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-08-13</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 2</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>4</td>
      <td>0</td>
      <td>Bournemouth</td>
      <td>...</td>
      <td>4-2-3-1</td>
      <td>David Coote</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>19</td>
      <td>7</td>
      <td>17.5</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-08-21</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 3</td>
      <td>Sun</td>
      <td>Away</td>
      <td>D</td>
      <td>3</td>
      <td>3</td>
      <td>Newcastle Utd</td>
      <td>...</td>
      <td>4-3-3</td>
      <td>Jarred Gillett</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>21</td>
      <td>10</td>
      <td>16.2</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-08-27</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 4</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>4</td>
      <td>2</td>
      <td>Crystal Palace</td>
      <td>...</td>
      <td>4-2-3-1</td>
      <td>Darren England</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>18</td>
      <td>5</td>
      <td>14.1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
years = list(range(2023, 2018, -1))
all_matches = []
```


```python
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
```


```python
import time
for year in years:
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text)
    standings_table = soup.select('table.stats_table')[0]

    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]
    
    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://fbref.com{previous_season}"
    
    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        data = requests.get(team_url)
        matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
        soup = BeautifulSoup(data.text)
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        data = requests.get(f"https://fbref.com{links[0]}")
        shooting = pd.read_html(data.text, match="Shooting")[0]
        shooting.columns = shooting.columns.droplevel()
        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue
        team_data = team_data[team_data["Comp"] == "Premier League"]
        
        team_data["Season"] = year
        team_data["Team"] = team_name
        all_matches.append(team_data)
        time.sleep(1)
```


```python
match_df = pd.concat(all_matches)
```


```python
match_df.columns = [c.lower() for c in match_df.columns]
```


```python
match_df.head()
```


```python
match_df.to_csv("matches.csv")
```


```python
matches = pd.read_csv("matches.csv", index_col=0)
```


```python
matches.shape
```




    (1520, 27)



### DATA CLEANING


```python
matches.dtypes
```




    date             object
    time             object
    comp             object
    round            object
    day              object
    venue            object
    result           object
    gf                int64
    ga                int64
    opponent         object
    xg              float64
    xga             float64
    poss            float64
    attendance      float64
    captain          object
    formation        object
    referee          object
    match report     object
    notes           float64
    sh              float64
    sot             float64
    dist            float64
    fk              float64
    pk              float64
    pkatt           float64
    season            int64
    team             object
    dtype: object




```python
matches["date"] = pd.to_datetime(matches["date"])
```

#### CREATE PREDICTORS


```python
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["team_code"] = matches["team"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(':.+', '', regex=True).astype('int')
matches["day_code"] = matches['date'].dt.dayofweek
# matches["target"] = matches['result'].astype('category').cat.codes
matches["target"] = (matches['result'] == 'W').astype('int')
matches.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>time</th>
      <th>comp</th>
      <th>round</th>
      <th>day</th>
      <th>venue</th>
      <th>result</th>
      <th>gf</th>
      <th>ga</th>
      <th>opponent</th>
      <th>...</th>
      <th>pk</th>
      <th>pkatt</th>
      <th>season</th>
      <th>team</th>
      <th>venue_code</th>
      <th>opp_code</th>
      <th>team_code</th>
      <th>hour</th>
      <th>day_code</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2022-08-07</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 1</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>2</td>
      <td>0</td>
      <td>West Ham</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2023</td>
      <td>Manchester City</td>
      <td>0</td>
      <td>21</td>
      <td>13</td>
      <td>16</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-08-13</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 2</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>4</td>
      <td>0</td>
      <td>Bournemouth</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2023</td>
      <td>Manchester City</td>
      <td>1</td>
      <td>2</td>
      <td>13</td>
      <td>15</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-08-21</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 3</td>
      <td>Sun</td>
      <td>Away</td>
      <td>D</td>
      <td>3</td>
      <td>3</td>
      <td>Newcastle Utd</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2023</td>
      <td>Manchester City</td>
      <td>0</td>
      <td>15</td>
      <td>13</td>
      <td>16</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-08-27</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 4</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>4</td>
      <td>2</td>
      <td>Crystal Palace</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2023</td>
      <td>Manchester City</td>
      <td>1</td>
      <td>7</td>
      <td>13</td>
      <td>15</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022-08-31</td>
      <td>19:30</td>
      <td>Premier League</td>
      <td>Matchweek 5</td>
      <td>Wed</td>
      <td>Home</td>
      <td>W</td>
      <td>6</td>
      <td>0</td>
      <td>Nott'ham Forest</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2023</td>
      <td>Manchester City</td>
      <td>1</td>
      <td>17</td>
      <td>13</td>
      <td>19</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>



### MODEL TRAINING


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
```


```python
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
```


```python
train = matches[matches["date"] < '2023-01-01']
test = matches[matches["date"] >= '2023-01-01']
```


```python
predictors = ['venue_code', 'opp_code', 'hour', 'day_code']
```


```python
rf.fit(train[predictors], train['target'])
```




    RandomForestClassifier(min_samples_split=10, n_estimators=50, random_state=1)




```python
preds = rf.predict(test[predictors])
```


```python
acc =accuracy_score(test['target'], preds)
acc
```




    0.5949074074074074




```python
combined = pd.DataFrame({'Actual':test['target'], 'Prediction':preds})
```


```python
pd.crosstab(index=combined['Actual'], columns=combined['Prediction'])
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Prediction</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Actual</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>204</td>
      <td>62</td>
    </tr>
    <tr>
      <th>1</th>
      <td>113</td>
      <td>53</td>
    </tr>
  </tbody>
</table>
</div>




```python
precision = precision_score(combined['Actual'], combined['Prediction'])
```

### MODEL TUNING


```python
grouped_matches = matches.groupby('team')
grouped_matches.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>time</th>
      <th>comp</th>
      <th>round</th>
      <th>day</th>
      <th>venue</th>
      <th>result</th>
      <th>gf</th>
      <th>ga</th>
      <th>opponent</th>
      <th>...</th>
      <th>pk</th>
      <th>pkatt</th>
      <th>season</th>
      <th>team</th>
      <th>venue_code</th>
      <th>opp_code</th>
      <th>team_code</th>
      <th>hour</th>
      <th>day_code</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2022-08-07</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 1</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>2</td>
      <td>0</td>
      <td>West Ham</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2023</td>
      <td>Manchester City</td>
      <td>0</td>
      <td>21</td>
      <td>13</td>
      <td>16</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-08-13</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 2</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>4</td>
      <td>0</td>
      <td>Bournemouth</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2023</td>
      <td>Manchester City</td>
      <td>1</td>
      <td>2</td>
      <td>13</td>
      <td>15</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-08-21</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 3</td>
      <td>Sun</td>
      <td>Away</td>
      <td>D</td>
      <td>3</td>
      <td>3</td>
      <td>Newcastle Utd</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2023</td>
      <td>Manchester City</td>
      <td>0</td>
      <td>15</td>
      <td>13</td>
      <td>16</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-08-27</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 4</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>4</td>
      <td>2</td>
      <td>Crystal Palace</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2023</td>
      <td>Manchester City</td>
      <td>1</td>
      <td>7</td>
      <td>13</td>
      <td>15</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022-08-31</td>
      <td>19:30</td>
      <td>Premier League</td>
      <td>Matchweek 5</td>
      <td>Wed</td>
      <td>Home</td>
      <td>W</td>
      <td>6</td>
      <td>0</td>
      <td>Nott'ham Forest</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2023</td>
      <td>Manchester City</td>
      <td>1</td>
      <td>17</td>
      <td>13</td>
      <td>19</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2021-08-14</td>
      <td>17:30</td>
      <td>Premier League</td>
      <td>Matchweek 1</td>
      <td>Sat</td>
      <td>Home</td>
      <td>L</td>
      <td>0</td>
      <td>3</td>
      <td>Liverpool</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Norwich City</td>
      <td>1</td>
      <td>12</td>
      <td>16</td>
      <td>17</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-08-21</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 2</td>
      <td>Sat</td>
      <td>Away</td>
      <td>L</td>
      <td>0</td>
      <td>5</td>
      <td>Manchester City</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Norwich City</td>
      <td>0</td>
      <td>13</td>
      <td>16</td>
      <td>15</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-08-28</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 3</td>
      <td>Sat</td>
      <td>Home</td>
      <td>L</td>
      <td>1</td>
      <td>2</td>
      <td>Leicester City</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2022</td>
      <td>Norwich City</td>
      <td>1</td>
      <td>11</td>
      <td>16</td>
      <td>15</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-09-11</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 4</td>
      <td>Sat</td>
      <td>Away</td>
      <td>L</td>
      <td>0</td>
      <td>1</td>
      <td>Arsenal</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Norwich City</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>15</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2021-09-18</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 5</td>
      <td>Sat</td>
      <td>Home</td>
      <td>L</td>
      <td>1</td>
      <td>3</td>
      <td>Watford</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Norwich City</td>
      <td>1</td>
      <td>20</td>
      <td>16</td>
      <td>15</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>115 rows × 33 columns</p>
</div>




```python
group = grouped_matches.get_group('Manchester City')
group
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>time</th>
      <th>comp</th>
      <th>round</th>
      <th>day</th>
      <th>venue</th>
      <th>result</th>
      <th>gf</th>
      <th>ga</th>
      <th>opponent</th>
      <th>...</th>
      <th>pk</th>
      <th>pkatt</th>
      <th>season</th>
      <th>team</th>
      <th>venue_code</th>
      <th>opp_code</th>
      <th>team_code</th>
      <th>hour</th>
      <th>day_code</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2022-08-07</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 1</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>2</td>
      <td>0</td>
      <td>West Ham</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2023</td>
      <td>Manchester City</td>
      <td>0</td>
      <td>21</td>
      <td>13</td>
      <td>16</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-08-13</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 2</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>4</td>
      <td>0</td>
      <td>Bournemouth</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2023</td>
      <td>Manchester City</td>
      <td>1</td>
      <td>2</td>
      <td>13</td>
      <td>15</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-08-21</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 3</td>
      <td>Sun</td>
      <td>Away</td>
      <td>D</td>
      <td>3</td>
      <td>3</td>
      <td>Newcastle Utd</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2023</td>
      <td>Manchester City</td>
      <td>0</td>
      <td>15</td>
      <td>13</td>
      <td>16</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-08-27</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 4</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>4</td>
      <td>2</td>
      <td>Crystal Palace</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2023</td>
      <td>Manchester City</td>
      <td>1</td>
      <td>7</td>
      <td>13</td>
      <td>15</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022-08-31</td>
      <td>19:30</td>
      <td>Premier League</td>
      <td>Matchweek 5</td>
      <td>Wed</td>
      <td>Home</td>
      <td>W</td>
      <td>6</td>
      <td>0</td>
      <td>Nott'ham Forest</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2023</td>
      <td>Manchester City</td>
      <td>1</td>
      <td>17</td>
      <td>13</td>
      <td>19</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>52</th>
      <td>2022-04-30</td>
      <td>17:30</td>
      <td>Premier League</td>
      <td>Matchweek 35</td>
      <td>Sat</td>
      <td>Away</td>
      <td>W</td>
      <td>4</td>
      <td>0</td>
      <td>Leeds United</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
      <td>0</td>
      <td>10</td>
      <td>13</td>
      <td>17</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>54</th>
      <td>2022-05-08</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 36</td>
      <td>Sun</td>
      <td>Home</td>
      <td>W</td>
      <td>5</td>
      <td>0</td>
      <td>Newcastle Utd</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
      <td>1</td>
      <td>15</td>
      <td>13</td>
      <td>16</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>55</th>
      <td>2022-05-11</td>
      <td>20:15</td>
      <td>Premier League</td>
      <td>Matchweek 33</td>
      <td>Wed</td>
      <td>Away</td>
      <td>W</td>
      <td>5</td>
      <td>1</td>
      <td>Wolves</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
      <td>0</td>
      <td>22</td>
      <td>13</td>
      <td>20</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2022-05-15</td>
      <td>14:00</td>
      <td>Premier League</td>
      <td>Matchweek 37</td>
      <td>Sun</td>
      <td>Away</td>
      <td>D</td>
      <td>2</td>
      <td>2</td>
      <td>West Ham</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2022</td>
      <td>Manchester City</td>
      <td>0</td>
      <td>21</td>
      <td>13</td>
      <td>14</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2022-05-22</td>
      <td>16:00</td>
      <td>Premier League</td>
      <td>Matchweek 38</td>
      <td>Sun</td>
      <td>Home</td>
      <td>W</td>
      <td>3</td>
      <td>2</td>
      <td>Aston Villa</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>16</td>
      <td>6</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>76 rows × 33 columns</p>
</div>



#### ADDING ROLLING AVERAGE (TAKING PAST 3 WEEKS DATA INTO CONSIDERATION)


```python
def rolling_averages(group, cols, new_cols):
    group = group.sort_values('date')
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group
```


```python
cols = ['gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt']
new_cols = [f"{c}_rolling" for c in cols]
```


```python
rolling_averages(group, cols, new_cols)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>time</th>
      <th>comp</th>
      <th>round</th>
      <th>day</th>
      <th>venue</th>
      <th>result</th>
      <th>gf</th>
      <th>ga</th>
      <th>opponent</th>
      <th>...</th>
      <th>day_code</th>
      <th>target</th>
      <th>gf_rolling</th>
      <th>ga_rolling</th>
      <th>sh_rolling</th>
      <th>sot_rolling</th>
      <th>dist_rolling</th>
      <th>fk_rolling</th>
      <th>pk_rolling</th>
      <th>pkatt_rolling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>2021-09-11</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 4</td>
      <td>Sat</td>
      <td>Away</td>
      <td>W</td>
      <td>1</td>
      <td>0</td>
      <td>Leicester City</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>3.333333</td>
      <td>0.333333</td>
      <td>19.666667</td>
      <td>6.000000</td>
      <td>16.866667</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2021-09-18</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 5</td>
      <td>Sat</td>
      <td>Home</td>
      <td>D</td>
      <td>0</td>
      <td>0</td>
      <td>Southampton</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>3.666667</td>
      <td>0.000000</td>
      <td>22.000000</td>
      <td>7.333333</td>
      <td>15.866667</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2021-09-25</td>
      <td>12:30</td>
      <td>Premier League</td>
      <td>Matchweek 6</td>
      <td>Sat</td>
      <td>Away</td>
      <td>W</td>
      <td>1</td>
      <td>0</td>
      <td>Chelsea</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>22.000000</td>
      <td>6.333333</td>
      <td>15.166667</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2021-10-03</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 7</td>
      <td>Sun</td>
      <td>Away</td>
      <td>D</td>
      <td>2</td>
      <td>2</td>
      <td>Liverpool</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>18.666667</td>
      <td>4.000000</td>
      <td>15.933333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2021-10-16</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 8</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>2</td>
      <td>0</td>
      <td>Burnley</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>1.000000</td>
      <td>0.666667</td>
      <td>14.333333</td>
      <td>2.333333</td>
      <td>16.833333</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>52</th>
      <td>2023-05-06</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 35</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>2</td>
      <td>1</td>
      <td>Leeds United</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>3.000000</td>
      <td>0.666667</td>
      <td>13.666667</td>
      <td>8.000000</td>
      <td>15.433333</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>54</th>
      <td>2023-05-14</td>
      <td>14:00</td>
      <td>Premier League</td>
      <td>Matchweek 36</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>3</td>
      <td>0</td>
      <td>Everton</td>
      <td>...</td>
      <td>6</td>
      <td>1</td>
      <td>2.333333</td>
      <td>0.666667</td>
      <td>14.666667</td>
      <td>7.000000</td>
      <td>16.366667</td>
      <td>0.666667</td>
      <td>0.333333</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2023-05-21</td>
      <td>16:00</td>
      <td>Premier League</td>
      <td>Matchweek 37</td>
      <td>Sun</td>
      <td>Home</td>
      <td>W</td>
      <td>1</td>
      <td>0</td>
      <td>Chelsea</td>
      <td>...</td>
      <td>6</td>
      <td>1</td>
      <td>2.666667</td>
      <td>0.333333</td>
      <td>14.000000</td>
      <td>5.666667</td>
      <td>18.100000</td>
      <td>1.333333</td>
      <td>0.000000</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2023-05-24</td>
      <td>20:00</td>
      <td>Premier League</td>
      <td>Matchweek 32</td>
      <td>Wed</td>
      <td>Away</td>
      <td>D</td>
      <td>1</td>
      <td>1</td>
      <td>Brighton</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>2.000000</td>
      <td>0.333333</td>
      <td>13.666667</td>
      <td>4.000000</td>
      <td>18.933333</td>
      <td>1.333333</td>
      <td>0.000000</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2023-05-28</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 38</td>
      <td>Sun</td>
      <td>Away</td>
      <td>L</td>
      <td>0</td>
      <td>1</td>
      <td>Brentford</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>1.666667</td>
      <td>0.333333</td>
      <td>12.333333</td>
      <td>3.333333</td>
      <td>17.333333</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>73 rows × 41 columns</p>
</div>




```python
matches_rolling = matches.groupby('team').apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>date</th>
      <th>time</th>
      <th>comp</th>
      <th>round</th>
      <th>day</th>
      <th>venue</th>
      <th>result</th>
      <th>gf</th>
      <th>ga</th>
      <th>opponent</th>
      <th>...</th>
      <th>day_code</th>
      <th>target</th>
      <th>gf_rolling</th>
      <th>ga_rolling</th>
      <th>sh_rolling</th>
      <th>sot_rolling</th>
      <th>dist_rolling</th>
      <th>fk_rolling</th>
      <th>pk_rolling</th>
      <th>pkatt_rolling</th>
    </tr>
    <tr>
      <th>team</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Arsenal</th>
      <th>4</th>
      <td>2021-09-11</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 4</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>1</td>
      <td>0</td>
      <td>Norwich City</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>9.666667</td>
      <td>2.333333</td>
      <td>14.833333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2021-09-18</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 5</td>
      <td>Sat</td>
      <td>Away</td>
      <td>W</td>
      <td>1</td>
      <td>0</td>
      <td>Burnley</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>0.333333</td>
      <td>2.333333</td>
      <td>12.333333</td>
      <td>3.000000</td>
      <td>14.133333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2021-09-26</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 6</td>
      <td>Sun</td>
      <td>Home</td>
      <td>W</td>
      <td>3</td>
      <td>1</td>
      <td>Tottenham</td>
      <td>...</td>
      <td>6</td>
      <td>1</td>
      <td>0.666667</td>
      <td>1.666667</td>
      <td>14.666667</td>
      <td>3.000000</td>
      <td>14.800000</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2021-10-02</td>
      <td>17:30</td>
      <td>Premier League</td>
      <td>Matchweek 7</td>
      <td>Sat</td>
      <td>Away</td>
      <td>D</td>
      <td>0</td>
      <td>0</td>
      <td>Brighton</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1.666667</td>
      <td>0.333333</td>
      <td>18.333333</td>
      <td>5.333333</td>
      <td>18.433333</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2021-10-18</td>
      <td>20:00</td>
      <td>Premier League</td>
      <td>Matchweek 8</td>
      <td>Mon</td>
      <td>Home</td>
      <td>D</td>
      <td>2</td>
      <td>2</td>
      <td>Crystal Palace</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1.333333</td>
      <td>0.333333</td>
      <td>11.000000</td>
      <td>4.000000</td>
      <td>19.833333</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Wolverhampton Wanderers</th>
      <th>39</th>
      <td>2023-04-29</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 34</td>
      <td>Sat</td>
      <td>Away</td>
      <td>L</td>
      <td>0</td>
      <td>6</td>
      <td>Brighton</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1.666667</td>
      <td>0.666667</td>
      <td>11.666667</td>
      <td>4.666667</td>
      <td>18.700000</td>
      <td>0.666667</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2023-05-06</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 35</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>1</td>
      <td>0</td>
      <td>Aston Villa</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>1.000000</td>
      <td>2.666667</td>
      <td>11.333333</td>
      <td>2.333333</td>
      <td>18.800000</td>
      <td>0.666667</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2023-05-13</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 36</td>
      <td>Sat</td>
      <td>Away</td>
      <td>L</td>
      <td>0</td>
      <td>2</td>
      <td>Manchester Utd</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>2.000000</td>
      <td>17.766667</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2023-05-20</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 37</td>
      <td>Sat</td>
      <td>Home</td>
      <td>D</td>
      <td>1</td>
      <td>1</td>
      <td>Everton</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0.333333</td>
      <td>2.666667</td>
      <td>7.000000</td>
      <td>1.333333</td>
      <td>15.600000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2023-05-28</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 38</td>
      <td>Sun</td>
      <td>Away</td>
      <td>L</td>
      <td>0</td>
      <td>5</td>
      <td>Arsenal</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>0.666667</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>2.333333</td>
      <td>15.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>1451 rows × 41 columns</p>
</div>




```python
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])
```


```python
matches_rolling
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>time</th>
      <th>comp</th>
      <th>round</th>
      <th>day</th>
      <th>venue</th>
      <th>result</th>
      <th>gf</th>
      <th>ga</th>
      <th>opponent</th>
      <th>...</th>
      <th>day_code</th>
      <th>target</th>
      <th>gf_rolling</th>
      <th>ga_rolling</th>
      <th>sh_rolling</th>
      <th>sot_rolling</th>
      <th>dist_rolling</th>
      <th>fk_rolling</th>
      <th>pk_rolling</th>
      <th>pkatt_rolling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-09-11</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 4</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>1</td>
      <td>0</td>
      <td>Norwich City</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>9.666667</td>
      <td>2.333333</td>
      <td>14.833333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-09-18</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 5</td>
      <td>Sat</td>
      <td>Away</td>
      <td>W</td>
      <td>1</td>
      <td>0</td>
      <td>Burnley</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>0.333333</td>
      <td>2.333333</td>
      <td>12.333333</td>
      <td>3.000000</td>
      <td>14.133333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-09-26</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 6</td>
      <td>Sun</td>
      <td>Home</td>
      <td>W</td>
      <td>3</td>
      <td>1</td>
      <td>Tottenham</td>
      <td>...</td>
      <td>6</td>
      <td>1</td>
      <td>0.666667</td>
      <td>1.666667</td>
      <td>14.666667</td>
      <td>3.000000</td>
      <td>14.800000</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-10-02</td>
      <td>17:30</td>
      <td>Premier League</td>
      <td>Matchweek 7</td>
      <td>Sat</td>
      <td>Away</td>
      <td>D</td>
      <td>0</td>
      <td>0</td>
      <td>Brighton</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1.666667</td>
      <td>0.333333</td>
      <td>18.333333</td>
      <td>5.333333</td>
      <td>18.433333</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-10-18</td>
      <td>20:00</td>
      <td>Premier League</td>
      <td>Matchweek 8</td>
      <td>Mon</td>
      <td>Home</td>
      <td>D</td>
      <td>2</td>
      <td>2</td>
      <td>Crystal Palace</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1.333333</td>
      <td>0.333333</td>
      <td>11.000000</td>
      <td>4.000000</td>
      <td>19.833333</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1446</th>
      <td>2023-04-29</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 34</td>
      <td>Sat</td>
      <td>Away</td>
      <td>L</td>
      <td>0</td>
      <td>6</td>
      <td>Brighton</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1.666667</td>
      <td>0.666667</td>
      <td>11.666667</td>
      <td>4.666667</td>
      <td>18.700000</td>
      <td>0.666667</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>1447</th>
      <td>2023-05-06</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 35</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>1</td>
      <td>0</td>
      <td>Aston Villa</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>1.000000</td>
      <td>2.666667</td>
      <td>11.333333</td>
      <td>2.333333</td>
      <td>18.800000</td>
      <td>0.666667</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>1448</th>
      <td>2023-05-13</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 36</td>
      <td>Sat</td>
      <td>Away</td>
      <td>L</td>
      <td>0</td>
      <td>2</td>
      <td>Manchester Utd</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>2.000000</td>
      <td>17.766667</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>1449</th>
      <td>2023-05-20</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 37</td>
      <td>Sat</td>
      <td>Home</td>
      <td>D</td>
      <td>1</td>
      <td>1</td>
      <td>Everton</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0.333333</td>
      <td>2.666667</td>
      <td>7.000000</td>
      <td>1.333333</td>
      <td>15.600000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1450</th>
      <td>2023-05-28</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 38</td>
      <td>Sun</td>
      <td>Away</td>
      <td>L</td>
      <td>0</td>
      <td>5</td>
      <td>Arsenal</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>0.666667</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>2.333333</td>
      <td>15.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>1451 rows × 41 columns</p>
</div>




```python
def make_predictions(data, predictors):
    train = data[data["date"] < '2023-01-01']
    test = data[data["date"] >= '2023-01-01']
    rf.fit(train[predictors], train['target'])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame({'Actual':test['target'], 'Prediction':preds})
    acc =accuracy_score(test['target'], preds)
    precision = precision_score(combined['Actual'], combined['Prediction'])
    return combined, acc, precision
```


```python
combined, acc2, precision2 = make_predictions(matches_rolling, predictors+new_cols)
```


```python
acc, acc2
```




    (0.5949074074074074, 0.6342592592592593)




```python
precision, precision2
```




    (0.4608695652173913, 0.5384615384615384)



### OBSERVATION: USING ROLLING AVERAGES AND NEW PREDICTORS IMPROVED BOTH ACCURACY AND PRECISION.


```python
combined = combined.merge(matches_rolling[['date','team', 'opponent', 'result', 'venue']], right_index=True, left_index=True)
combined
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Prediction</th>
      <th>date</th>
      <th>team</th>
      <th>opponent</th>
      <th>result</th>
      <th>venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>51</th>
      <td>0</td>
      <td>1</td>
      <td>2023-01-03</td>
      <td>Arsenal</td>
      <td>Newcastle Utd</td>
      <td>D</td>
      <td>Home</td>
    </tr>
    <tr>
      <th>52</th>
      <td>1</td>
      <td>0</td>
      <td>2023-01-15</td>
      <td>Arsenal</td>
      <td>Tottenham</td>
      <td>W</td>
      <td>Away</td>
    </tr>
    <tr>
      <th>53</th>
      <td>1</td>
      <td>1</td>
      <td>2023-01-22</td>
      <td>Arsenal</td>
      <td>Manchester Utd</td>
      <td>W</td>
      <td>Home</td>
    </tr>
    <tr>
      <th>54</th>
      <td>0</td>
      <td>1</td>
      <td>2023-02-04</td>
      <td>Arsenal</td>
      <td>Everton</td>
      <td>L</td>
      <td>Away</td>
    </tr>
    <tr>
      <th>55</th>
      <td>0</td>
      <td>1</td>
      <td>2023-02-11</td>
      <td>Arsenal</td>
      <td>Brentford</td>
      <td>D</td>
      <td>Home</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1446</th>
      <td>0</td>
      <td>0</td>
      <td>2023-04-29</td>
      <td>Wolverhampton Wanderers</td>
      <td>Brighton</td>
      <td>L</td>
      <td>Away</td>
    </tr>
    <tr>
      <th>1447</th>
      <td>1</td>
      <td>0</td>
      <td>2023-05-06</td>
      <td>Wolverhampton Wanderers</td>
      <td>Aston Villa</td>
      <td>W</td>
      <td>Home</td>
    </tr>
    <tr>
      <th>1448</th>
      <td>0</td>
      <td>0</td>
      <td>2023-05-13</td>
      <td>Wolverhampton Wanderers</td>
      <td>Manchester Utd</td>
      <td>L</td>
      <td>Away</td>
    </tr>
    <tr>
      <th>1449</th>
      <td>0</td>
      <td>0</td>
      <td>2023-05-20</td>
      <td>Wolverhampton Wanderers</td>
      <td>Everton</td>
      <td>D</td>
      <td>Home</td>
    </tr>
    <tr>
      <th>1450</th>
      <td>0</td>
      <td>0</td>
      <td>2023-05-28</td>
      <td>Wolverhampton Wanderers</td>
      <td>Arsenal</td>
      <td>L</td>
      <td>Away</td>
    </tr>
  </tbody>
</table>
<p>432 rows × 7 columns</p>
</div>



#### IMPROVING DATA CONSISTENCY FOR ANALYSIS


```python
class MissingDict(dict):
    __missing__ = lambda self, key: key
    
team_map = {}
for i, team in enumerate(sorted(combined['team'].unique())):
    team_map[team] = sorted(combined['opponent'].unique())[i]
mapping = MissingDict(**team_map)
```


```python
combined['new_team'] = combined['team'].map(mapping)
```


```python
combined
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Prediction</th>
      <th>date</th>
      <th>team</th>
      <th>opponent</th>
      <th>result</th>
      <th>venue</th>
      <th>new_team</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>51</th>
      <td>0</td>
      <td>1</td>
      <td>2023-01-03</td>
      <td>Arsenal</td>
      <td>Newcastle Utd</td>
      <td>D</td>
      <td>Home</td>
      <td>Arsenal</td>
    </tr>
    <tr>
      <th>52</th>
      <td>1</td>
      <td>0</td>
      <td>2023-01-15</td>
      <td>Arsenal</td>
      <td>Tottenham</td>
      <td>W</td>
      <td>Away</td>
      <td>Arsenal</td>
    </tr>
    <tr>
      <th>53</th>
      <td>1</td>
      <td>1</td>
      <td>2023-01-22</td>
      <td>Arsenal</td>
      <td>Manchester Utd</td>
      <td>W</td>
      <td>Home</td>
      <td>Arsenal</td>
    </tr>
    <tr>
      <th>54</th>
      <td>0</td>
      <td>1</td>
      <td>2023-02-04</td>
      <td>Arsenal</td>
      <td>Everton</td>
      <td>L</td>
      <td>Away</td>
      <td>Arsenal</td>
    </tr>
    <tr>
      <th>55</th>
      <td>0</td>
      <td>1</td>
      <td>2023-02-11</td>
      <td>Arsenal</td>
      <td>Brentford</td>
      <td>D</td>
      <td>Home</td>
      <td>Arsenal</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1446</th>
      <td>0</td>
      <td>0</td>
      <td>2023-04-29</td>
      <td>Wolverhampton Wanderers</td>
      <td>Brighton</td>
      <td>L</td>
      <td>Away</td>
      <td>Wolves</td>
    </tr>
    <tr>
      <th>1447</th>
      <td>1</td>
      <td>0</td>
      <td>2023-05-06</td>
      <td>Wolverhampton Wanderers</td>
      <td>Aston Villa</td>
      <td>W</td>
      <td>Home</td>
      <td>Wolves</td>
    </tr>
    <tr>
      <th>1448</th>
      <td>0</td>
      <td>0</td>
      <td>2023-05-13</td>
      <td>Wolverhampton Wanderers</td>
      <td>Manchester Utd</td>
      <td>L</td>
      <td>Away</td>
      <td>Wolves</td>
    </tr>
    <tr>
      <th>1449</th>
      <td>0</td>
      <td>0</td>
      <td>2023-05-20</td>
      <td>Wolverhampton Wanderers</td>
      <td>Everton</td>
      <td>D</td>
      <td>Home</td>
      <td>Wolves</td>
    </tr>
    <tr>
      <th>1450</th>
      <td>0</td>
      <td>0</td>
      <td>2023-05-28</td>
      <td>Wolverhampton Wanderers</td>
      <td>Arsenal</td>
      <td>L</td>
      <td>Away</td>
      <td>Wolves</td>
    </tr>
  </tbody>
</table>
<p>432 rows × 8 columns</p>
</div>




```python
merged = combined.merge(combined, left_on=['date', 'team'], right_on=['date', 'opponent'])
merged
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual_x</th>
      <th>Prediction_x</th>
      <th>date</th>
      <th>team_x</th>
      <th>opponent_x</th>
      <th>result_x</th>
      <th>venue_x</th>
      <th>new_team_x</th>
      <th>Actual_y</th>
      <th>Prediction_y</th>
      <th>team_y</th>
      <th>opponent_y</th>
      <th>result_y</th>
      <th>venue_y</th>
      <th>new_team_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2023-01-03</td>
      <td>Arsenal</td>
      <td>Newcastle Utd</td>
      <td>D</td>
      <td>Home</td>
      <td>Arsenal</td>
      <td>0</td>
      <td>0</td>
      <td>Newcastle United</td>
      <td>Arsenal</td>
      <td>D</td>
      <td>Away</td>
      <td>Newcastle Utd</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>2023-01-15</td>
      <td>Arsenal</td>
      <td>Tottenham</td>
      <td>W</td>
      <td>Away</td>
      <td>Arsenal</td>
      <td>0</td>
      <td>0</td>
      <td>Tottenham Hotspur</td>
      <td>Arsenal</td>
      <td>L</td>
      <td>Home</td>
      <td>Tottenham</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>2023-01-22</td>
      <td>Arsenal</td>
      <td>Manchester Utd</td>
      <td>W</td>
      <td>Home</td>
      <td>Arsenal</td>
      <td>0</td>
      <td>1</td>
      <td>Manchester United</td>
      <td>Arsenal</td>
      <td>L</td>
      <td>Away</td>
      <td>Manchester Utd</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>2023-02-04</td>
      <td>Arsenal</td>
      <td>Everton</td>
      <td>L</td>
      <td>Away</td>
      <td>Arsenal</td>
      <td>1</td>
      <td>0</td>
      <td>Everton</td>
      <td>Arsenal</td>
      <td>W</td>
      <td>Home</td>
      <td>Everton</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>2023-02-11</td>
      <td>Arsenal</td>
      <td>Brentford</td>
      <td>D</td>
      <td>Home</td>
      <td>Arsenal</td>
      <td>0</td>
      <td>0</td>
      <td>Brentford</td>
      <td>Arsenal</td>
      <td>D</td>
      <td>Away</td>
      <td>Brentford</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>276</th>
      <td>0</td>
      <td>0</td>
      <td>2023-04-30</td>
      <td>Southampton</td>
      <td>Newcastle Utd</td>
      <td>L</td>
      <td>Away</td>
      <td>Southampton</td>
      <td>1</td>
      <td>1</td>
      <td>Newcastle United</td>
      <td>Southampton</td>
      <td>W</td>
      <td>Home</td>
      <td>Newcastle Utd</td>
    </tr>
    <tr>
      <th>277</th>
      <td>0</td>
      <td>0</td>
      <td>2023-05-08</td>
      <td>Southampton</td>
      <td>Nott'ham Forest</td>
      <td>L</td>
      <td>Away</td>
      <td>Southampton</td>
      <td>1</td>
      <td>0</td>
      <td>Nottingham Forest</td>
      <td>Southampton</td>
      <td>W</td>
      <td>Home</td>
      <td>Nott'ham Forest</td>
    </tr>
    <tr>
      <th>278</th>
      <td>0</td>
      <td>0</td>
      <td>2023-05-13</td>
      <td>Southampton</td>
      <td>Fulham</td>
      <td>L</td>
      <td>Home</td>
      <td>Southampton</td>
      <td>1</td>
      <td>0</td>
      <td>Fulham</td>
      <td>Southampton</td>
      <td>W</td>
      <td>Away</td>
      <td>Fulham</td>
    </tr>
    <tr>
      <th>279</th>
      <td>0</td>
      <td>0</td>
      <td>2023-05-21</td>
      <td>Southampton</td>
      <td>Brighton</td>
      <td>L</td>
      <td>Away</td>
      <td>Southampton</td>
      <td>1</td>
      <td>0</td>
      <td>Brighton and Hove Albion</td>
      <td>Southampton</td>
      <td>W</td>
      <td>Home</td>
      <td>Brighton</td>
    </tr>
    <tr>
      <th>280</th>
      <td>0</td>
      <td>0</td>
      <td>2023-05-28</td>
      <td>Southampton</td>
      <td>Liverpool</td>
      <td>D</td>
      <td>Home</td>
      <td>Southampton</td>
      <td>0</td>
      <td>1</td>
      <td>Liverpool</td>
      <td>Southampton</td>
      <td>D</td>
      <td>Away</td>
      <td>Liverpool</td>
    </tr>
  </tbody>
</table>
<p>281 rows × 15 columns</p>
</div>




```python
merged[(merged['Prediction_x'] == 1) & (merged['Prediction_y'] == 0)]['Actual_x'].value_counts()
```




    Actual_x
    1    32
    0    24
    Name: count, dtype: int64




```python

```
