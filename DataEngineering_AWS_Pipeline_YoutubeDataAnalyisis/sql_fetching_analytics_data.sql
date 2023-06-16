-- Extract the necessary data
SELECT 
    DISTINCT (a.video_id), 
    a.title, 
    a.channel_title, 
    a.category_id,
    a.snippet_title,
    a.trending_date,
    a.publish_time,
    a.views,
    a.likes,
    a.dislikes,
    a.comment_count,
    a.description,
    a.region,
    CASE
        WHEN a.region='gb' THEN 'Great Britain'
        WHEN a.region='us' THEN 'United States'
        WHEN a.region='ca' THEN 'Canada'
    END AS Country
FROM 
    db_dataengg_youtube_analytics_tonmoy.final_analytics_data a
ORDER BY
    a.video_id

-- Group Data by Published YEar for Analysis

SELECT
    a.publish_time, 
    YEAR(CAST(SUBSTR(a.publish_time, 1, 10) as TIMESTAMP)) published_year,
    MONTH(CAST(SUBSTR(a.publish_time, 1, 10) as TIMESTAMP)) published_month,
    COUNT(DISTINCT (a.video_id)) no_videos, 
    SUM(a.views) total_views,
    SUM(a.likes) total_likes,
    SUM(a.dislikes) total_dislikes,
    SUM(a.comment_count) total_comments
FROM 
    db_dataengg_youtube_analytics_tonmoy.final_analytics_data a
GROUP BY
    YEAR(CAST(SUBSTR(a.publish_time, 1, 10) as TIMESTAMP)), 
    MONTH(CAST(SUBSTR(a.publish_time, 1, 10) as TIMESTAMP)),
    a.publish_time