import os
import csv
import re
from googleapiclient.discovery import build

api_key = 'AIzaSyCCWuhJcsGlZ5pnRLhyWYbtaMGYwVmdEsY'
youtube = build('youtube', 'v3', developerKey=api_key)

def get_video_id_from_url(url):
    video_id = re.search(r'v=([a-zA-Z0-9_-]+)', url)
    return video_id.group(1) if video_id else None

def get_video_title(youtube, video_id):
    request = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()
    return request['items'][0]['snippet']['title']

def scrape_video_comments(video_url):
    video_id = get_video_id_from_url(video_url)
    if not video_id:
        return None, None
    
    video_title = get_video_title(youtube, video_id)
    csv_filename = os.path.join('mydata', f"{video_title}.csv")
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['COMMENT_ID', 'AUTHOR', 'DATE', 'CONTENT']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100
        ).execute()
        
        comments = []
        
        while response:
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                # Remove '@' symbol from the author's name
                AUTHOR = comment['authorDisplayName'].replace('@', '')
                writer.writerow({
                    'COMMENT_ID': item['id'],
                    'AUTHOR': comment['authorDisplayName'],
                    'DATE': comment['publishedAt'],
                    'CONTENT': comment['textDisplay']
                })
                comments.append({
                    'COMMENT_ID': item['id'],
                    'AUTHOR': comment['authorDisplayName'],
                    'DATE': comment['publishedAt'],
                    'CONTENT': comment['textDisplay']
                })
            if 'nextPageToken' in response:
                response = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    pageToken=response['nextPageToken'],
                    maxResults=100
                ).execute()
            else:
                break
        
        return csv_filename, comments


