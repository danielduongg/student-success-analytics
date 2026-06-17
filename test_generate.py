from generate_data import generate

def test_columns_and_target():
    df = generate(n=500, seed=1)
    assert {"retained_to_year2", "first_gen", "first_term_gpa"}.issubset(df.columns)
    assert df.retained_to_year2.isin([0, 1]).all()

def test_retention_rate_realistic():
    df = generate(n=3000, seed=2)
    assert 0.68 < df.retained_to_year2.mean() < 0.84

def test_engagement_signal_present():
    # higher LMS engagement should associate with higher retention
    df = generate(n=4000, seed=3)
    lo = df[df.weekly_lms_logins <= 3].retained_to_year2.mean()
    hi = df[df.weekly_lms_logins >= 9].retained_to_year2.mean()
    assert hi > lo
