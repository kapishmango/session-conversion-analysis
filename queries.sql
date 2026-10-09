-- These queries use the training partition; test labels do not inform EDA.
SELECT VisitorType, COUNT(*) AS sessions, SUM(Revenue) AS purchases,
       ROUND(AVG(Revenue), 4) AS conversion_rate
FROM sessions WHERE split = 'train'
GROUP BY VisitorType ORDER BY sessions DESC;

SELECT Month, COUNT(*) AS sessions, SUM(Revenue) AS purchases,
       ROUND(AVG(Revenue), 4) AS conversion_rate
FROM sessions WHERE split = 'train'
GROUP BY Month;

SELECT Revenue, COUNT(*) AS sessions,
       ROUND(AVG(ProductRelated), 2) AS mean_product_pages,
       ROUND(AVG(ProductRelated_Duration), 2) AS mean_product_seconds
FROM sessions WHERE split = 'train'
GROUP BY Revenue;
