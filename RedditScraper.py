import time;
import httpx;
import pandas as pd;

# Basic Extraction

base_url = 'https://www.reddit.com'
subreddits = ['HealthyFood', 'nutrition', 'organic', 'Milk', 'eggs', 'meat', 'food', 'cooking', 'Agriculture']
keywords = ["organic", "natural", "egg", "milk", "dairy", "meat", "beef", "chicken", "gmo", "genetically modified", "livestock"]

SUBREDDIT = 'HealthyFood'

endpoint = f'/r/{SUBREDDIT}'

url = base_url + endpoint + '/search.json'
dataset = []

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
}

for keyword in keywords:
    after_post_id = None

    for _ in range(5):
        params = {
            'q': keyword,
            'restrict_sr': 1,
            'sort': 'top',
            't': 'all',
            'limit': 100,
            'after': after_post_id
        }
        response = httpx.get(url, params=params, headers=headers)
        print(f'fetching "{response.url}"...')
        if response.status_code != 200:
            raise Exception(f'Failed to fetch data: {response.status_code}')
        
        json_data = response.json()

        children = json_data['data']['children']
        if not children:
            break

        for rec in children:
            post_data = rec['data']
            post_data['query_keyword'] = keyword
            dataset.append(post_data)

        after_post_id = json_data['data']['after']
        if after_post_id is None:
            break

        time.sleep(0.5)

df = pd.DataFrame(dataset)
if not df.empty and 'id' in df.columns:
    df = df.drop_duplicates(subset=['id'])
df.to_csv(f'reddit_{SUBREDDIT}.csv', index = False)