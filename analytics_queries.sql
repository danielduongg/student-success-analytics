-- Analytics queries for the student-retention warehouse table `students`.
-- Run them with: python src/analytics.py  (loads CSV into in-memory DuckDB)

-- Q1: Retention rate and cohort size by entry year
SELECT entry_cohort,
       count(*)                              AS cohort_size,
       round(avg(retained_to_year2) * 100, 1) AS retention_pct
FROM students
GROUP BY entry_cohort
ORDER BY entry_cohort;

-- Q2: Retention by first-generation status
SELECT CASE WHEN first_gen = 1 THEN 'First-gen' ELSE 'Continuing-gen' END AS student_type,
       count(*)                               AS n,
       round(avg(retained_to_year2) * 100, 1) AS retention_pct
FROM students
GROUP BY first_gen
ORDER BY retention_pct DESC;

-- Q3: Retention by engagement quartile (weekly LMS logins)
WITH q AS (
  SELECT weekly_lms_logins, retained_to_year2,
         ntile(4) OVER (ORDER BY weekly_lms_logins) AS engagement_quartile
  FROM students
)
SELECT engagement_quartile,
       min(weekly_lms_logins) AS min_logins,
       max(weekly_lms_logins) AS max_logins,
       round(avg(retained_to_year2) * 100, 1) AS retention_pct
FROM q
GROUP BY engagement_quartile
ORDER BY engagement_quartile;

-- Q4: Highest-risk majors by dropout rate (min cohort of 100)
SELECT major,
       count(*) AS n,
       round((1 - avg(retained_to_year2)) * 100, 1) AS dropout_pct
FROM students
GROUP BY major
HAVING count(*) >= 100
ORDER BY dropout_pct DESC
LIMIT 5;

-- Q5: Unmet financial need — retained vs not
SELECT CASE WHEN retained_to_year2 = 1 THEN 'Retained' ELSE 'Dropped' END AS outcome,
       round(avg(unmet_need_usd), 0) AS avg_unmet_need_usd,
       round(avg(first_term_gpa), 2) AS avg_first_term_gpa
FROM students
GROUP BY retained_to_year2;
