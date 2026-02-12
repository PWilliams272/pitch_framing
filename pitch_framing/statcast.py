import requests
import pandas as pd
import pybaseball
from bs4 import BeautifulSoup
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

STATCAST_REQUEST_HEADERS = {
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.8',
    'priority': 'u=1, i',
    'referer': 'https://baseballsavant.mlb.com/statcast_search?hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea=2025%7C&hfSit=&player_type=pitcher&hfOuts=&hfOpponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&hfMo=&hfTeam=&home_road=&hfRO=&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=&hfFlag=&metric_1=&group_by=name&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc',
    'sec-ch-ua': '"Brave";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'sec-gpc': '1',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
    'x-requested-with': 'XMLHttpRequest',
}

class StatCast:
    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = Path.home() / ".pitch_framing"
        else:
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        pybaseball.cache.enable()

    def _fetch_video_urls(self, start_date, end_date, pitcher_id):
        """Helper function to submit HTTP request to the statcast page containing video URLs of individual plays
        """
        fn = f"video_url_dataframes/{start_date}_{end_date}_{pitcher_id}.parquet"
        if self.cache_dir and (self.cache_dir / fn).exists():
            return pd.read_parquet(self.cache_dir / fn)

        start_season = int(start_date[:4])
        end_season = int(end_date[:4])
        season_list = [str(year) for year in range(start_season, end_season + 1)]
        params = {
            'hfSea': '|'.join(season_list),  # Season
            'player_type': 'pitcher',  # pitcher
            'game_date_gt': start_date,
            'game_date_lt': end_date,
            'group_by': 'name',
            'player_event_sort': 'api_p_release_speed',
            'sort_order': 'desc',
            'type': 'details',
            'player_id': pitcher_id
        }
        response = requests.get(
            'https://baseballsavant.mlb.com/statcast_search',
            params=params,
            headers=STATCAST_REQUEST_HEADERS
        )
        df = self._process_video_url_response(response)
        if self.cache_dir:
            os.makedirs(self.cache_dir / "video_url_dataframes", exist_ok=True)
            df.to_parquet(self.cache_dir / fn)
        return df

    @staticmethod
    def _process_video_url_response(response):
        """Parses the HTML response the statcast video search page
        """
        soup = BeautifulSoup(response.text, 'html.parser')

        plays = []
        for row in soup.find_all('tr'):
            cells = row.find_all('td')
            if not cells or len(cells) < 15:
                continue
            cell_map = {
                0: 'date',
                1: 'pitch_type',
                2: 'pitch_mph',
                3: 'pitch_spin_rate',
                4: 'pitcher_name',
                5: 'batter_name',
                6: 'exit_velocity_mph',
                7: 'launch_angle_deg',
                8: 'distance_ft',
                9: 'zone',
            }
            if len(cells) == 15:
                cell_map.update({
                    10: 'count',
                    11: 'inning',
                    12: 'pitch_result',
                    13: 'pa_result',
                    14: 'video_url'
                })
            elif len(cells) == 16:
                cell_map.update({
                    10: 'series_game',
                    11: 'count',
                    12: 'inning',
                    13: 'pitch_result',
                    14: 'pa_result',
                    15: 'video_url'
                })
            data = {
                v: cells[k].get_text(strip=True) for k, v in cell_map.items()
            }
            # Find video link
            data['video_url'] = None
            video_cell = cells[-1]
            a_tag = video_cell.find('a', href=True)
            if a_tag and 'playId=' in a_tag['href']:
                data['video_url'] = 'https://baseballsavant.mlb.com' + a_tag['href']
            plays.append(data)

        df = pd.DataFrame(plays)
        return df

    def get_video_urls(self, start_date, end_date, pitcher_id):
        """Get the video url data as a DataFrame for a specific pitcher
        """
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        # Chunk by month-long increments
        video_urls = []
        _start_date = start_date_dt.replace(day=1)
        while _start_date < end_date_dt:
            _end_date = (_start_date + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
            _end_date = min(_end_date, pd.to_datetime('today'))
            df = self._fetch_video_urls(
                _start_date.strftime('%Y-%m-%d'),
                _end_date.strftime('%Y-%m-%d'),
                pitcher_id
            )
            video_urls.append(df)
            _start_date += pd.DateOffset(months=1)
        df = pd.concat(video_urls, ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= start_date_dt) & (df['date'] <= end_date_dt)]
        df['pitcher_id'] = pitcher_id
        return df

    def get_data(self, start_date, end_date):
        """Pulls in statcast data, video URL data, and merges into one DataFrame
        """
        df_statcast = pybaseball.statcast(start_date, end_date)
        pitcher_id_list = list(df_statcast['pitcher'].unique())

        # Use multithreading to speed up, limited by network calls
        results = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(self.get_video_urls, start_date, end_date, pitcher_id): pitcher_id for pitcher_id in pitcher_id_list}
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    results.append(future.result())
                except Exception as e:
                    pitcher_id = futures[future]
                    print(f"Pitcher {pitcher_id} generated an exception: {e}")
        df_video_urls = pd.concat(results, ignore_index=True)
        
        """
        There's no play id/pitch id available so we can't match the two dataframes
        with 100% confidence. The combo of date, inning, count, pitch_type, pitch_mph,
        pitch_spin_rate, zone should be *nearly* unique, so we'll use that for merging
        and skip any duplicates.
        """
        merge_cols = ['pitcher', 'game_date', 'inning', 'balls', 'strikes', 'pitch_type',
                        'release_speed', 'release_spin_rate', 'zone']
        df_video_urls = df_video_urls.rename(columns={
            'date': 'game_date',
            'pitch_mph': 'release_speed',
            'pitch_spin_rate': 'release_spin_rate'
        })
        df_video_urls['pitcher'] = df_video_urls['pitcher_id']
        df_video_urls['balls'] = df_video_urls['count'].apply(lambda x: int(x.split('-')[0]) if '-' in x else 0)
        df_video_urls['strikes'] = df_video_urls['count'].apply(lambda x: int(x.split('-')[1]) if '-' in x else 0)
        df_video_urls['release_spin_rate'] = df_video_urls['release_spin_rate'].apply(lambda x: None if x in ['--', ''] else float(x.replace(',', '')))
        df_video_urls['release_speed'] = df_video_urls['release_speed'].apply(lambda x: None if x in ['--', ''] else float(x))
        df_video_urls['zone'] = df_video_urls['zone'].apply(lambda x: None if x in ['--', ''] else float(x))
        df_video_urls['inning'] = df_video_urls['inning'].apply(lambda x: int(x[5:].strip()))

        # Remove duplicates and add back on later
        duplicates = df_statcast[df_statcast.duplicated(subset=merge_cols, keep=False)]
        df_statcast = df_statcast[~df_statcast.duplicated(subset=merge_cols, keep=False)]

        # Remove automatic balls (easy duplicates)
        df_video_urls = df_video_urls[~df_video_urls['pitch_type'].isin(['AB', 'IBB'])]

        # Drop duplicates on merge_cols
        df_video_urls = df_video_urls.drop_duplicates(subset=merge_cols)

        df_statcast = pd.merge(
            df_statcast,
            df_video_urls[merge_cols + ['video_url']],
            on=merge_cols,
            how='left',
            validate='1:1'
        )
        df_statcast = pd.concat([df_statcast, duplicates], ignore_index=True)

        return df_statcast